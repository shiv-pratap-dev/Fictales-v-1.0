# app/main.py
"""
Fictales Railway core (final, UI-friendly)

Generate-story UI expectations (Swagger/form):
- file (UploadFile)        -> kid's image
- kid_name (string)        -> text input
- story_choice (dropdown)  -> "story1" or "story2"
- gender (dropdown)        -> "boy" or "girl"

Pipeline (5-page checkpoint build, preview-first):
  1) upload kid image to uploads bucket
  2) build kidinfo = {kid_name, gender, kid_image_r2}
  3) list templates for story_choice (cap to 5 pages)
  4) For each template:
       - text-swap (TEXT_PIAPI or DZINE) with retries + validation
       - if page <= 3: face-swap immediately, upload preview
       - else: store for face-swap pass later
  5) after all face-swaps: upload pages, assemble PDF, upload final PDF
"""
import os
import uuid
import json
import logging
import base64
import time
from datetime import datetime
from typing import Optional, Dict, Any, List, Literal, Tuple
import boto3
from botocore.client import Config
import httpx
from fastapi import FastAPI, BackgroundTasks, HTTPException, UploadFile, File, Form
from pydantic import BaseModel

from PIL import Image, UnidentifiedImageError
from io import BytesIO

# ---------- logging ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fictales-railway")

# ---------- env / config ----------
CF_ACCOUNT_ID = os.getenv("CF_ACCOUNT_ID")
CF_R2_BUCKET_UPLOADS = os.getenv("CF_R2_BUCKET_UPLOADS", "fictales-uploads")
CF_R2_BUCKET_OUTPUTS = os.getenv("CF_R2_BUCKET_OUTPUTS", "fictales-outputs")
CF_R2_BUCKET_TEMPLATES = os.getenv("CF_R2_BUCKET_TEMPLATES", "fictales-templates")

CF_ACCESS_KEY_ID = os.getenv("CF_ACCESS_KEY_ID")
CF_SECRET_ACCESS_KEY = os.getenv("CF_SECRET_ACCESS_KEY")

# FACE-SWAP service (existing account)
PIAPI_URL = os.getenv("PIAPI_URL")
PIAPI_KEY = os.getenv("PIAPI_KEY")

# TEXT-SWAP service (separate account)
TEXT_PIAPI_URL = os.getenv("TEXT_PIAPI_URL")
TEXT_PIAPI_KEY = os.getenv("TEXT_PIAPI_KEY")

# Optional fallback (Dzine)
DZINE_URL = os.getenv("DZINE_URL")

PRESIGN_EXPIRATION = int(os.getenv("PRESIGN_EXPIRATION", "3600"))

R2_ENDPOINT = f"https://{CF_ACCOUNT_ID}.r2.cloudflarestorage.com" if CF_ACCOUNT_ID else None

# ---------- s3 / r2 client ----------
s3 = None
if R2_ENDPOINT and CF_ACCESS_KEY_ID and CF_SECRET_ACCESS_KEY:
    try:
        s3 = boto3.client(
            "s3",
            endpoint_url=R2_ENDPOINT,
            aws_access_key_id=CF_ACCESS_KEY_ID,
            aws_secret_access_key=CF_SECRET_ACCESS_KEY,
            config=Config(signature_version="s3v4"),
        )
        logger.info("Initialized S3 client for R2 at %s", R2_ENDPOINT)
    except Exception as e:
        logger.exception("Failed to init S3 client: %s", e)
        s3 = None
else:
    logger.warning("R2 not fully configured (endpoint or creds missing).")

# ---------- app ----------
app = FastAPI(title="FictalesRailwayCore (Final UI)")

# simple in-memory job store (replace with persistent store for prod)
_jobs: Dict[str, Dict[str, Any]] = {}

# ---------- helper models ----------
class PresignRequest(BaseModel):
    filename: str
    content_type: Optional[str] = None

class PresignResp(BaseModel):
    upload_url: str
    r2_key: str
    bucket: str
    expires_in: int

# ---------- helpers ----------
def make_r2_key(prefix: str, filename: str) -> str:
    safe = filename.replace(" ", "_")
    return f"{prefix}/{uuid.uuid4().hex}_{safe}"

def generate_presigned_put(bucket: str, key: str, content_type: Optional[str] = None, expires_in: int = PRESIGN_EXPIRATION) -> str:
    if s3 is None:
        raise RuntimeError("S3 client not configured")
    params = {"Bucket": bucket, "Key": key}
    if content_type:
        params["ContentType"] = content_type
    return s3.generate_presigned_url(ClientMethod="put_object", Params=params, ExpiresIn=expires_in)

def generate_presigned_get(bucket: str, key: str, expires_in: int = PRESIGN_EXPIRATION) -> str:
    if s3 is None:
        raise RuntimeError("S3 client not configured")
    params = {"Bucket": bucket, "Key": key}
    return s3.generate_presigned_url(ClientMethod="get_object", Params=params, ExpiresIn=expires_in)

def s3_put_bytes(bucket: str, key: str, data: bytes, content_type: Optional[str] = None):
    if s3 is None:
        raise RuntimeError("S3 client not configured")
    kwargs = {"Bucket": bucket, "Key": key, "Body": data}
    if content_type:
        kwargs["ContentType"] = content_type
    s3.put_object(**kwargs)

def s3_get_bytes(bucket: str, key: str) -> bytes:
    if s3 is None:
        raise RuntimeError("S3 client not configured")
    resp = s3.get_object(Bucket=bucket, Key=key)
    return resp["Body"].read()

def s3_list_keys(bucket: str, prefix: str) -> List[str]:
    if s3 is None:
        raise RuntimeError("S3 client not configured")
    keys = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    return keys

def s3_object_exists(bucket: str, key: str) -> bool:
    if s3 is None:
        return False
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False

def natural_sort_key(name: str):
    base = os.path.basename(name)
    parts = base.split(".")[0].split("_")
    for p in reversed(parts):
        if p.isdigit():
            return int(p)
    return base

def is_image_key(key: str) -> bool:
    lower = key.lower()
    return lower.endswith(".png") or lower.endswith(".jpg") or lower.endswith(".jpeg") or lower.endswith(".webp")

def is_valid_image_bytes(b: bytes) -> bool:
    """Return True if bytes represent a valid/openable image."""
    try:
        with Image.open(BytesIO(b)) as img:
            img.verify()  # will raise if not a valid image
        return True
    except Exception:
        return False

# ---------- external API helpers (models hardcoded) ----------
# NOTE: keep these functions as "dumb" callers — we will wrap them with safety wrappers
async def call_text_piapi_image_swap(image_bytes: bytes, kidinfo: dict, template_meta: dict = None, timeout: int = 120) -> bytes:
    """
    Text-swap step using PiAPI Gemini (image-to-image variant).
    This function attempts the primary PiAPI flow and returns raw bytes (which may be non-image).
    Higher-level wrapper 'safe_text_swap' will validate and retry as needed.
    """
    import asyncio

    if not TEXT_PIAPI_URL:
        raise RuntimeError("TEXT_PIAPI_URL not configured")

    url = TEXT_PIAPI_URL.rstrip("/") + "/"
    headers = {
        "X-API-KEY": TEXT_PIAPI_KEY,
        "User-Agent": "Fictales/1.0 (FastAPI client)",
        "Accept": "application/json",
        "Connection": "keep-alive"
    }

    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    kid_name = kidinfo.get("kid_name", "child")

    prompt = (
        f"Replace only the name text 'shiv' with '{kid_name}'. "
        "Preserve the same font, font size, color, shadow, alignment, and layout exactly as it appears. "
        "Do not modify other text or visuals."
    )

    payload = {
        "model": "gemini",
        "task_type": "gemini-2.5-flash-image",
        "input": {
            "prompt": prompt,
            "image": f"data:image/png;base64,{image_b64}",
            "num_images": 1,
            "output_format": "png",
            "aspect_ratio": "1:1"
        }
    }

    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        # primary POST
        r = None
        try:
            r = await client.post(url, headers=headers, json=payload)
            if r.status_code in (301, 302, 303, 307, 308):
               logger.info("%s TEXT API redirected to: %s", repr(template_meta), r.headers.get('Location'))
        except Exception as e:
            logger.warning("[%s] TEXT_PIAPI POST failed: %s", repr(template_meta), e)
            r = None

        # return raw bytes only if we can (the caller will validate)
        if r is None:
            raise HTTPException(status_code=502, detail="TEXT_PIAPI not reachable")
        # handle immediate non-JSON (like redirects to HTML error pages)
        if r.status_code >= 400:
            raise HTTPException(status_code=502, detail=f"Gemini POST returned {r.status_code}")

        # If Gemini returns an immediate completed response in JSON, attempt to pull image
        try:
            resp = r.json()
        except Exception:
            # non-json (or empty) - return raw content to be validated by caller
            return r.content

        data = resp.get("data", {}) or {}
        status = data.get("status") or ""
        if status == "completed":
            out = data.get("output", {}) or {}
            if out.get("image_b64"):
                return base64.b64decode(out["image_b64"])
            urls = out.get("image_urls") or []
            if urls:
                # return the raw bytes from first url
                resp_get = await client.get(urls[0])
                resp_get.raise_for_status()
                return resp_get.content
            # completed but no image -> raise
            raise HTTPException(status_code=502, detail="Gemini completed but returned no image")

        # otherwise we will poll for task_id
        task_id = data.get("task_id") or ""
        if not task_id:
            raise HTTPException(status_code=502, detail="Gemini did not return task_id")

        task_status_url = TEXT_PIAPI_URL.rstrip("/") + f"/{task_id}"
        poll_interval = 1.0
        max_poll_seconds = min(max(10, timeout), 120)
        max_attempts = int(max_poll_seconds / poll_interval)

        for attempt in range(max_attempts):
            await asyncio.sleep(poll_interval)
            try:
                r2 = await client.get(task_status_url, headers=headers)
            except Exception as e:
                logger.warning("[%s] Poll attempt %d failed: %s", task_id, attempt + 1, e)
                continue

            if r2.status_code >= 500:
                logger.warning("[%s] Gemini poll returned %s; attempt %d/%d", task_id, r2.status_code, attempt + 1, max_attempts)
                continue
            if r2.status_code >= 400:
                raise HTTPException(status_code=502, detail=f"Gemini poll error: {r2.status_code}")

            try:
                jr = r2.json()
            except Exception:
                # If poll response is not JSON, check raw content
                if r2.content:
                    return r2.content
                continue

            d = jr.get("data", {}) or {}
            st = d.get("status", "")
            if st == "completed":
                out = d.get("output", {}) or {}
                if out.get("image_b64"):
                    return base64.b64decode(out["image_b64"])
                urls = out.get("image_urls") or []
                if urls:
                    resp_get = await client.get(urls[0])
                    resp_get.raise_for_status()
                    return resp_get.content
                raise HTTPException(status_code=502, detail="Gemini completed but returned no image")
            if st in ("failed", "error"):
                raise HTTPException(status_code=502, detail="Gemini processing failed")

        raise HTTPException(status_code=504, detail="Gemini processing timed out")


async def call_face_piapi_swap(image_bytes: bytes, face_meta: dict = None, timeout: int = 180) -> bytes:
    """
    Call PIAPI (Image-Toolkit) to do face swapping on a single image.
    Returns raw bytes; caller will validate.
    """
    if PIAPI_URL:
        headers = {"Authorization": f"Bearer {PIAPI_KEY}"} if PIAPI_KEY else {}
        data = {
            "model": "Qubico/image-toolkit",
            "task_type": "img2img",
            "input": {
                "meta": face_meta or {}
            }
        }
        files = {"file": ("page.png", image_bytes, "image/png")}
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            try:
                r = await client.post(PIAPI_URL, headers=headers, data={"payload": json.dumps(data)}, files=files)
            except Exception as e:
                logger.warning("[face-swap] Failed to reach PIAPI image-toolkit: %s", e)
                raise HTTPException(status_code=502, detail="PIAPI not reachable")

            if r.status_code >= 400:
                raise HTTPException(status_code=502, detail=f"PIAPI returned {r.status_code}")

            # try to parse json and prefer image_b64
            try:
                j = r.json()
                if isinstance(j, dict):
                    if j.get("image_b64"):
                        return base64.b64decode(j["image_b64"])
                    if j.get("output_url"):
                        resp = await client.get(j["output_url"])
                        resp.raise_for_status()
                        return resp.content
            except Exception:
                pass
            return r.content

    # If PIAPI_URL not configured, raise so caller can fallback to DZINE
    raise HTTPException(status_code=502, detail="PIAPI_URL not configured")


# ---------- safety wrappers (validate + retry + fallback) ----------
async def safe_text_swap(tpl_bytes: bytes, kidinfo: dict, template_meta: dict = None, timeout: int = 120) -> bytes:
    """
    Robust wrapper:
      - try call_text_piapi_image_swap up to 3 times
      - validate bytes are real images
      - if still invalid and DZINE_URL present, try Dzine once
      - final degraded fallback: return original template bytes
    """
    # Try primary PiAPI up to 3 times
    for attempt in range(3):
        try:
            logger.info("[%s] text-swap attempt %d/3", repr(template_meta), attempt + 1)
            out = await call_text_piapi_image_swap(tpl_bytes, kidinfo, template_meta=template_meta, timeout=timeout)
            if out and is_valid_image_bytes(out):
                return out
            logger.warning("[%s] text-swap returned invalid image on attempt %d", repr(template_meta), attempt + 1)
        except HTTPException as he:
            logger.warning("[%s] text-swap HTTP error on attempt %d: %s", repr(template_meta), attempt + 1, he.detail)
        except Exception as e:
            logger.warning("[%s] text-swap exception on attempt %d: %s", repr(template_meta), attempt + 1, e)
        await asyncio_sleep_backoff(attempt)

    # Try DZINE fallback (if configured)
    if DZINE_URL:
        try:
            logger.info("[%s] Trying DZINE fallback for text-swap", repr(template_meta))
            async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
                files = {"file": ("page.png", tpl_bytes, "image/png")}
                data = {"meta": json.dumps({"prompt": f"Replace name with {kidinfo.get('kid_name', 'child')}", "num_images": 1})}
                dz_r = await client.post(DZINE_URL, data=data, files=files)
                if dz_r.status_code < 400:
                    # try parse json
                    try:
                        j = dz_r.json()
                        if isinstance(j, dict) and j.get("image_b64"):
                            out = base64.b64decode(j["image_b64"])
                            if is_valid_image_bytes(out):
                                return out
                        if isinstance(j, dict) and j.get("output_url"):
                            resp = await client.get(j["output_url"])
                            resp.raise_for_status()
                            if is_valid_image_bytes(resp.content):
                                return resp.content
                    except Exception:
                        # maybe raw bytes
                        if is_valid_image_bytes(dz_r.content):
                            return dz_r.content
                else:
                    logger.warning("[%s] DZINE fallback returned status %s", repr(template_meta), dz_r.status_code)
        except Exception as e:
            logger.exception("[%s] DZINE fallback failed: %s", repr(template_meta), e)

    # Last resort: log and return original template bytes (degraded path)
    logger.error("[%s] Both TEXT_PIAPI and fallback failed; returning original template bytes.", repr(template_meta))
    return tpl_bytes

async def safe_face_swap(image_bytes: bytes, face_meta: dict = None, timeout: int = 180) -> bytes:
    """
    Robust wrapper for face-swap:
      - try PIAPI face-swap upto 2 times
      - validate bytes
      - fallback to DZINE if configured
      - final: return original bytes
    """
    # Try PIAPI attempts
    for attempt in range(2):
        try:
            logger.info("[face-swap] attempt %d/2 for %s", attempt + 1, repr(face_meta))
            out = await call_face_piapi_swap(image_bytes, face_meta=face_meta, timeout=timeout)
            if out and is_valid_image_bytes(out):
                return out
            logger.warning("[face-swap] returned invalid image on attempt %d for %s", attempt + 1, repr(face_meta))
        except HTTPException as he:
            logger.warning("[face-swap] HTTP error attempt %d: %s", attempt + 1, he.detail)
        except Exception as e:
            logger.warning("[face-swap] Exception attempt %d: %s", attempt + 1, e)
        await asyncio_sleep_backoff(attempt)

    # DZINE fallback
    if DZINE_URL:
        try:
            logger.info("[face-swap] Trying DZINE fallback for face-swap %s", repr(face_meta))
            async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
                files = {"file": ("page.png", image_bytes, "image/png")}
                data = {"meta": json.dumps(face_meta or {})}
                dz_r = await client.post(DZINE_URL, data=data, files=files)
                if dz_r.status_code < 400:
                    try:
                        j = dz_r.json()
                        if isinstance(j, dict) and j.get("image_b64"):
                            out = base64.b64decode(j["image_b64"])
                            if is_valid_image_bytes(out):
                                return out
                        if isinstance(j, dict) and j.get("output_url"):
                            resp = await client.get(j["output_url"])
                            resp.raise_for_status()
                            if is_valid_image_bytes(resp.content):
                                return resp.content
                    except Exception:
                        if is_valid_image_bytes(dz_r.content):
                            return dz_r.content
                else:
                    logger.warning("[face-swap] DZINE returned %s", dz_r.status_code)
        except Exception as e:
            logger.exception("[face-swap] DZINE fallback failed: %s", e)

    logger.error("[face-swap] Both PIAPI and fallback failed; returning original image bytes.")
    return image_bytes

async def asyncio_sleep_backoff(attempt: int):
    """Simple async backoff helper."""
    import asyncio
    await asyncio.sleep(1.0 * (attempt + 1))

# ---------- endpoints ----------
@app.post("/get-presigned-upload", response_model=PresignResp)
async def get_presigned_upload(req: PresignRequest):
    if s3 is None:
        raise HTTPException(status_code=500, detail="R2 not configured")
    key = make_r2_key("uploads", req.filename)
    try:
        url = generate_presigned_put(CF_R2_BUCKET_UPLOADS, key, content_type=req.content_type)
    except Exception as e:
        logger.exception("Presign generation failed: %s", e)
        raise HTTPException(status_code=500, detail="Failed to generate presigned URL")
    return PresignResp(upload_url=url, r2_key=key, bucket=CF_R2_BUCKET_UPLOADS, expires_in=PRESIGN_EXPIRATION)

@app.post("/generate-story")
async def generate_story(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),  # kid image
    kid_name: str = Form(...),     # text input
    story_choice: Literal["story1", "story2"] = Form(...),  # dropdown
    gender: Literal["boy", "girl"] = Form(...),             # dropdown
):
    """
    Accepts kid image + kid_name + story_choice + gender.
    Uploads kid image to uploads bucket, constructs kidinfo, and starts pipeline.
    Returns job_id immediately. Frontend should poll /job-status/{job_id}.
    """
    if s3 is None:
        raise HTTPException(status_code=500, detail="Server storage not configured")

    job_id = uuid.uuid4().hex
    logger.info("Creating job %s for story=%s gender=%s kid=%s", job_id, story_choice, gender, kid_name)

    # read file
    try:
        content = await file.read()
    except Exception as e:
        logger.exception("[%s] failed reading uploaded file: %s", job_id, e)
        raise HTTPException(status_code=400, detail="Failed to read uploaded file")

    # upload kid image to uploads bucket
    upload_key = f"uploads/{job_id}_kid_{file.filename.replace(' ', '_')}"
    try:
        s3_put_bytes(CF_R2_BUCKET_UPLOADS, upload_key, content, content_type=file.content_type or "image/png")
        logger.info("[%s] uploaded kid image to R2: %s", job_id, upload_key)
    except Exception as e:
        logger.exception("[%s] Failed uploading kid image to R2: %s", job_id, e)
        raise HTTPException(status_code=500, detail="Failed to upload kid image to R2")

    # build kidinfo
    kidinfo = {
        "kid_name": kid_name,
        "gender": gender,
        "kid_image_r2": {"bucket": CF_R2_BUCKET_UPLOADS, "r2_key": upload_key}
    }

    # create job record with preview fields
    _jobs[job_id] = {
        "status": "processing",
        "created_at": datetime.utcnow().isoformat(),
        "error": None,
        "r2_key": None,
        "preview_ready": False,
        "preview_urls": [],
        "pages_total": 0
    }

    # start background pipeline
    background_tasks.add_task(_process_pipeline, job_id, upload_key, kidinfo, story_choice, gender)

    return {"job_id": job_id, "status": "processing"}

@app.get("/job-status/{job_id}")
async def job_status(job_id: str):
    data = _jobs.get(job_id)
    if not data:
        raise HTTPException(status_code=404, detail="job_id not found")
    result = {"job_id": job_id, "status": data.get("status", "processing"), "error": data.get("error")}
    result["preview_ready"] = data.get("preview_ready", False)
    result["preview_urls"] = data.get("preview_urls", [])
    if data.get("status") == "completed" and data.get("r2_key"):
        try:
            download_url = generate_presigned_get(CF_R2_BUCKET_OUTPUTS, data["r2_key"])
        except Exception:
            download_url = None
        result.update({"download_url": download_url, "r2_key": data.get("r2_key")})
    return result

@app.get("/health")
async def health():
    return {"ok": True, "time": datetime.utcnow().isoformat()}

@app.get("/test-piapi")
async def test_piapi():
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get("https://api.piapi.ai/api/v1/task")
            return {"status": r.status_code, "text": (r.text or "")[:400]}
    except Exception as e:
        return {"error": str(e)}

# ---------- background pipeline (preview-first) ----------
async def _process_pipeline(job_id: str, kidinfo_r2_key: str, kidinfo: dict, story_choice: str, gender: str):
    """
    Steps:
      1) list templates (cap to 5)
      2) text-swap ALL pages but immediately face-swap and publish preview for first 3
      3) face-swap remaining pages
      4) upload sequential pages and assemble final PDF
    """
    try:
        prefix = f"{story_choice}/{gender}/"
        logger.info("[%s] Template prefix resolved to: %s", job_id, prefix)

        logger.info("[%s] listing templates at %s/%s", job_id, CF_R2_BUCKET_TEMPLATES, prefix)
        try:
            keys = s3_list_keys(CF_R2_BUCKET_TEMPLATES, prefix)
        except Exception as e:
            logger.exception("[%s] failed listing templates: %s", job_id, e)
            _jobs[job_id].update(status="failed", error="Failed listing template files")
            return

        # Filter to images and natural sort
        image_keys = [k for k in keys if is_image_key(k)]
        if not image_keys:
            msg = f"No image templates found for {story_choice}/{gender} in templates bucket"
            logger.error("[%s] %s", job_id, msg)
            _jobs[job_id].update(status="failed", error=msg)
            return
        keys_sorted = sorted(image_keys, key=lambda k: natural_sort_key(k))

        # Cap to 5 pages for this checkpoint build
        keys_sorted = keys_sorted[:5]
        logger.info("[%s] using first %d template pages", job_id, len(keys_sorted))
        _jobs[job_id]["pages_total"] = len(keys_sorted)

        tmp_dir = f"/tmp/fictales_{job_id}"
        os.makedirs(tmp_dir, exist_ok=True)

        # containers to hold image bytes (final)
        final_image_bytes: List[Tuple[str, bytes]] = []  # (page_num, bytes)

        # ---------- PHASE A: TEXT-SWAP for all pages, face-swap immediately for first 3 ----------
        for idx, tpl_key in enumerate(keys_sorted, start=1):
            logger.info("[%s] [TEXT] downloading template %s", job_id, tpl_key)
            try:
                tpl_bytes = s3_get_bytes(CF_R2_BUCKET_TEMPLATES, tpl_key)
            except Exception as e:
                logger.exception("[%s] failed downloading template %s: %s", job_id, tpl_key, e)
                _jobs[job_id].update(status="failed", error=f"Failed downloading template {tpl_key}")
                return

            # robust text-swap -> validated image bytes
            try:
                text_swapped_bytes = await safe_text_swap(tpl_bytes, kidinfo, template_meta={"template_key": tpl_key, "index": idx})
            except Exception as e:
                logger.exception("[%s] TEXT swap totally failed for %s: %s", job_id, tpl_key, e)
                # final degraded path: use original template as text-swap
                text_swapped_bytes = tpl_bytes

            page_num = str(idx).zfill(3)

            # Save interim text-swapped locally & upload (debug)
            text_local_path = os.path.join(tmp_dir, f"text_{page_num}.png")
            with open(text_local_path, "wb") as f:
                f.write(text_swapped_bytes)
            try:
                tmp_r2_key = f"outputs/{job_id}/_textswap/{page_num}.png"
                s3_put_bytes(CF_R2_BUCKET_OUTPUTS, tmp_r2_key, text_swapped_bytes, content_type="image/png")
                logger.info("[%s] uploaded interim text-swapped -> %s", job_id, tmp_r2_key)
            except Exception as e:
                logger.warning("[%s] failed uploading interim text page (non-fatal): %s", job_id, e)

            # For first 3 pages, do face-swap immediately to produce preview-ready images
            if idx <= 3:
                try:
                    face_bytes = await safe_face_swap(text_swapped_bytes, face_meta={"kidinfo_key": kidinfo_r2_key, "index": idx})
                except Exception as e:
                    logger.exception("[%s] face-swap failed for preview page %s: %s", job_id, page_num, e)
                    face_bytes = text_swapped_bytes  # degraded

                # Validate final preview bytes
                if not is_valid_image_bytes(face_bytes):
                    logger.warning("[%s] preview page %s result invalid, using text-swapped bytes", job_id, page_num)
                    face_bytes = text_swapped_bytes

                # write preview local & upload preview to R2
                preview_local = os.path.join(tmp_dir, f"preview_{page_num}.png")
                with open(preview_local, "wb") as f:
                    f.write(face_bytes)

                preview_r2_key = f"outputs/{job_id}/preview/{page_num}.png"
                try:
                    s3_put_bytes(CF_R2_BUCKET_OUTPUTS, preview_r2_key, face_bytes, content_type="image/png")
                    logger.info("[%s] uploaded preview page -> %s", job_id, preview_r2_key)
                except Exception as e:
                    logger.warning("[%s] failed uploading preview page (non-fatal): %s", job_id, e)

                # Also save to final_image_bytes for PDF assembling
                final_image_bytes.append((page_num, face_bytes))

                # If we have first 3 previews, set preview_ready and store presigned URLs
                if idx == 3:
                    # generate presigned urls for the three preview images (if s3 configured)
                    preview_urls = []
                    try:
                        for j in range(1, 4):
                            key_j = f"outputs/{job_id}/preview/{str(j).zfill(3)}.png"
                            url = generate_presigned_get(CF_R2_BUCKET_OUTPUTS, key_j)
                            preview_urls.append(url)
                    except Exception as e:
                        logger.warning("[%s] failed generating preview presigned URLs: %s", job_id, e)
                    _jobs[job_id].update(preview_ready=True, preview_urls=preview_urls)
                    logger.info("[%s] First 3 pages ready — preview available.", job_id)
            else:
                # For pages > 3, store text-swapped bytes to be face-swapped in PHASE B
                final_image_bytes.append((page_num, text_swapped_bytes))

        # At this point: first 3 final_image_bytes are fully face-swapped,
        # remaining items may be text-swapped (need face-swap pass)
        # >>>>>>> REQUIRED CHECKPOINT LOG <<<<<<<
        logger.info("[%s] done with text-swap pass, proceeding to face-swap remaining pages", job_id)

        # ---------- PHASE B: FACE-SWAP remaining pages (pages > 3) ----------
        completed_final = []
        for page_num, img_bytes in final_image_bytes:
            # skip those already face-swapped (pages 001-003 we treated as final)
            if int(page_num) <= 3:
                completed_final.append((page_num, img_bytes))
                continue

            try:
                face_bytes = await safe_face_swap(img_bytes, face_meta={"kidinfo_key": kidinfo_r2_key, "index": int(page_num)})
            except Exception as e:
                logger.exception("[%s] face-swap failed for page %s: %s", job_id, page_num, e)
                face_bytes = img_bytes  # degraded

            # validate
            if not is_valid_image_bytes(face_bytes):
                logger.warning("[%s] final page %s invalid after face-swap; using text-swapped bytes", job_id, page_num)
                face_bytes = img_bytes

            # write local and upload final page
            local_path = os.path.join(tmp_dir, f"{page_num}.png")
            with open(local_path, "wb") as f:
                f.write(face_bytes)
            r2_out_key = f"outputs/{job_id}/{page_num}.png"
            try:
                s3_put_bytes(CF_R2_BUCKET_OUTPUTS, r2_out_key, face_bytes, content_type="image/png")
                logger.info("[%s] uploaded final page -> %s", job_id, r2_out_key)
            except Exception as e:
                logger.exception("[%s] failed uploading page to outputs: %s", job_id, e)
                _jobs[job_id].update(status="failed", error="Failed uploading page image to outputs")
                return

            completed_final.append((page_num, face_bytes))

        # Make sure we included the previewed pages in uploads (some were already uploaded as preview)
        # Upload any previewed pages to final path too (ensure consistent outputs)
        for page_num, img_bytes in completed_final:
            # if preview pages (<=3), also ensure outputs/{job_id}/NNN.png exist
            r2_out_key = f"outputs/{job_id}/{page_num}.png"
            try:
                s3_put_bytes(CF_R2_BUCKET_OUTPUTS, r2_out_key, img_bytes, content_type="image/png")
            except Exception as e:
                logger.warning("[%s] failed ensuring final upload for %s: %s", job_id, r2_out_key, e)

        # ---------- PHASE C: ASSEMBLE PDF ONCE ----------
        pdf_path = os.path.join(tmp_dir, f"{job_id}.pdf")
        try:
            images = []
            # sort completed_final by page number ascending
            completed_final_sorted = sorted(completed_final, key=lambda t: int(t[0]))
            for pnum, bytes_img in completed_final_sorted:
                try:
                    img = Image.open(BytesIO(bytes_img))
                    # convert if necessary
                    if img.mode in ("RGBA", "LA") or (img.mode == "P"):
                        img = img.convert("RGB")
                    images.append(img.copy())
                    img.close()
                except UnidentifiedImageError as ui:
                    logger.error("[%s] PIL failed to open page %s: %s", job_id, pnum, ui)
                except Exception as e:
                    logger.exception("[%s] unexpected error opening page %s: %s", job_id, pnum, e)

            if not images:
                raise RuntimeError("No valid images to assemble into PDF")

            first, rest = images[0], images[1:]
            first.save(pdf_path, "PDF", resolution=150.0, save_all=True, append_images=rest)
            logger.info("[%s] assembled PDF -> %s", job_id, pdf_path)
        except Exception as e:
            logger.exception("[%s] failed assembling PDF: %s", job_id, e)
            _jobs[job_id].update(status="failed", error="Failed assembling PDF")
            return

        final_pdf_r2_key = f"outputs/{job_id}/{job_id}.pdf"
        try:
            with open(pdf_path, "rb") as f:
                s3_put_bytes(CF_R2_BUCKET_OUTPUTS, final_pdf_r2_key, f.read(), content_type="application/pdf")
            _jobs[job_id].update(status="completed", r2_key=final_pdf_r2_key)
            logger.info("[%s] uploaded final PDF -> %s", job_id, final_pdf_r2_key)
            return
        except Exception as e:
            logger.exception("[%s] failed uploading final PDF: %s", job_id, e)
            _jobs[job_id].update(status="failed", error="Failed uploading final PDF to outputs")
            return

    except Exception as exc:
        logger.exception("[%s] unexpected error in pipeline: %s", job_id, exc)
        _jobs[job_id].update(status="failed", error=str(exc))
        return

# ---------- runner ----------
if __name__ == "__main__":
    import uvicorn, os
    port = int(os.environ.get("PORT", 8000))
    # Railway likes 8080; use the environment PORT when available
    uvicorn.run("app.main:app", host="0.0.0.0", port=port)
