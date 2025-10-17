# app/main.py
"""
Fictales Railway core (final, UI-friendly)

Generate-story UI expectations (Swagger/form):
- file (UploadFile)        -> kid's image
- kid_name (string)        -> text input
- story_choice (dropdown)  -> "story1" or "story2"
- gender (dropdown)        -> "boy" or "girl"

Pipeline (5-page check build):
  1) upload kid image to uploads bucket
  2) build kidinfo = {kid_name, gender, kid_image_r2}
  3) list templates for story_choice (cap to 5 pages)
  4) PHASE A: per-page text-swap for first 5 pages -> store temp results
     (after all 5 text-swaps) -> LOG checkpoint and proceed
  5) PHASE B: per-page face-swap over text-swapped results
  6) upload sequential pages and assemble single PDF once
  7) upload final PDF to outputs/{job_id}/{job_id}.pdf
"""
import os
import uuid
import json
import logging
import base64
from datetime import datetime
from typing import Optional, Dict, Any, List, Literal
import boto3
from botocore.client import Config
import httpx
from fastapi import FastAPI, BackgroundTasks, HTTPException, UploadFile, File, Form
from pydantic import BaseModel

from PIL import Image
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

# Optional fallback
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

# ---------- external API helpers (models hardcoded) ----------
async def call_text_piapi_image_swap(image_bytes: bytes, kidinfo: dict, template_meta: dict = None, timeout: int = 120) -> bytes:
    """
    Text-swap step using PiAPI Gemini (image-to-image variant).
    - Posts image+prompt to TEXT_PIAPI_URL (expects /api/v1/task)
    - Polls the returned task_id until 'completed' (or fails)
    - Returns final image bytes (decoded from image_b64 or fetched from image_urls[0])
    - Retries transient 5xx errors and falls back to DZINE_URL or original image if needed
    """
    import asyncio, time

    if not TEXT_PIAPI_URL:
        raise RuntimeError("TEXT_PIAPI_URL not configured")

    url = TEXT_PIAPI_URL.rstrip("/") + "/"
    headers = {
        "X-API-KEY": TEXT_PIAPI_KEY,
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Fictales/1.0 (compatible; Python; FastAPI)",
        "Accept": "application/json",
        "Connection": "keep-alive"
    }

    image_b64_payload = base64.b64encode(image_bytes).decode("utf-8")
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
            "image": f"data:image/png;base64,{image_b64_payload}",
            "num_images": 1,
            "output_format": "png",
            "aspect_ratio": "1:1"
        }
    }

    # Primary attempt with retries on transient 5xx (Cloudflare / origin flakiness)
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        try:
            r = await client.post(url, headers=headers, json=payload)
            # helpful debug if redirected
            if r.status_code in (301, 302, 303, 307, 308):
               logger.warning("%s TEXT API redirected to: %s", repr(template_meta), r.headers.get('Location'))
        except Exception as e:
            logger.exception("Failed to reach TEXT_PIAPI endpoint: %s", e)
            # Attempt fallback provider (DZINE) below before failing completely

            r = None

        # --- retry transient 5xx (esp. Cloudflare 502) ---
        if r is not None:
            for attempt in range(3):
                if r.status_code >= 500:
                    logger.warning("[%s] PiAPI returned %s, retry %d/3 ...", repr(template_meta), r.status_code, attempt + 1)
                    await asyncio.sleep(2 * (attempt + 1))  # simple backoff
                    try:
                        r = await client.post(url, headers=headers, json=payload)
                    except Exception as e:
                        logger.warning("[%s] retry attempt %d failed to reach TEXT_PIAPI: %s", repr(template_meta), attempt+1, e)
                        r = None
                        break
                else:
                    break
        # --- end retry block ---

        # If still no response or final 5xx, attempt DZINE fallback (if configured)
        if r is None or (r.status_code >= 500):
            if DZINE_URL:
                logger.info("[%s] Falling back to DZINE_URL for text-swap", repr(template_meta))
                try:
                    # DZINE expects form upload; adapt to its expected interface
                    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as dz_client:
                        files = {"file": ("page.png", image_bytes, "image/png")}
                        data = {"meta": json.dumps({"prompt": prompt, "num_images": 1})}
                        dz_r = await dz_client.post(DZINE_URL, data=data, files=files)
                        if dz_r.status_code >= 400:
                            logger.error("[%s] DZINE fallback returned %s", repr(template_meta), dz_r.status_code)
                        else:
                            try:
                                j = dz_r.json()
                                # prefer base64
                                if isinstance(j, dict):
                                    if j.get("image_b64"):
                                        return base64.b64decode(j["image_b64"])
                                    if j.get("output_url"):
                                        resp_get = await dz_client.get(j["output_url"])
                                        resp_get.raise_for_status()
                                        return resp_get.content
                            except Exception:
                                # if DZINE returns raw image bytes
                                return dz_r.content
                except Exception as e:
                    logger.exception("[%s] DZINE fallback failed: %s", repr(template_meta), e)

            # final degraded fallback: return original template bytes (so pipeline continues)
            logger.error("[%s] Both TEXT_PIAPI and fallback failed; returning original image bytes to avoid job failure.", repr(template_meta))
            return image_bytes

        # At this point we have a response 'r' from primary that should be JSON (or completed)
        text_lower = (r.text or "").lower()
        if r.status_code in (400, 402) and "credit" in text_lower:
            logger.error("Nano Banana / Gemini credits exhausted or quota reached: %s", (r.text or "")[:400])
            raise HTTPException(status_code=402, detail="Nano Banana credits exhausted. Please top up and retry.")

        if r.status_code >= 400:
            logger.error("Gemini POST returned %s: %s", r.status_code, (r.text or "")[:800])
            raise HTTPException(status_code=502, detail=f"Gemini error: {r.status_code}")

        try:
            resp = r.json()
        except Exception as e:
            logger.warning("Unable to parse JSON from Gemini POST response: %s", e)
            raise HTTPException(status_code=502, detail="Invalid response from Gemini")

        data = resp.get("data", {}) if isinstance(resp, dict) else {}
        task_id = data.get("task_id") or ""
        status = data.get("status") or ""

        # If instant completed
        if status == "completed":
            output = data.get("output", {}) or {}
            # always prefer raw b64; higher quality than ephemeral URL
            if output.get("image_b64"):
                return base64.b64decode(output["image_b64"])
            urls = output.get("image_urls") or []
            if urls:
                clean_url = urls[0]
                if "img.theapi.app/ephemeral" in clean_url:
                    logger.warning("[%s] Ephemeral CDN image detected; quality may be reduced. URL: %s", repr(template_meta), clean_url)
                resp_get = await client.get(clean_url)
                resp_get.raise_for_status()
                return resp_get.content
            raise HTTPException(status_code=502, detail="Gemini returned completed status but no image found")

        # Otherwise poll task endpoint
        if not task_id:
            logger.error("No task_id returned by Gemini POST: %s", resp)
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
                logger.warning("[%s] Poll attempt %d failed to reach Gemini: %s", task_id, attempt + 1, e)
                continue

            text2 = (r2.text or "").lower()
            if r2.status_code in (400, 402) and "credit" in text2:
                logger.error("[%s] Credits exhausted while polling: %s", task_id, (r2.text or "")[:400])
                raise HTTPException(status_code=402, detail="Nano Banana credits exhausted during processing")

            if r2.status_code >= 500:
                # transient poll error -> log and continue; do not fail job immediately
                logger.warning("[%s] Gemini poll returned %s; attempt %d/%d", task_id, r2.status_code, attempt + 1, max_attempts)
                continue

            if r2.status_code >= 400:
                logger.error("[%s] Gemini poll returned %s: %s", task_id, r2.status_code, (r2.text or "")[:800])
                raise HTTPException(status_code=502, detail=f"Gemini poll error: {r2.status_code}")

            try:
                jr = r2.json()
            except Exception as e:
                logger.warning("[%s] Failed parsing Gemini poll JSON (attempt %d): %s", task_id, attempt + 1, e)
                continue

            d = jr.get("data", {}) if isinstance(jr, dict) else {}
            st = d.get("status", "")
            if st == "completed":
                out = d.get("output", {}) or {}
                if out.get("image_b64"):
                    return base64.b64decode(out["image_b64"])
                urls = out.get("image_urls") or []
                if urls:
                    clean_url = urls[0]
                    if "img.theapi.app/ephemeral" in clean_url:
                        logger.warning("[%s] Ephemeral CDN image detected; quality may be reduced. URL: %s", task_id, clean_url)
                    resp_get = await client.get(clean_url)
                    resp_get.raise_for_status()
                    return resp_get.content
                raise HTTPException(status_code=502, detail="Gemini completed but returned no image")
            if st in ("failed", "error"):
                logger.error("[%s] Gemini task failed: %s", task_id, jr)
                raise HTTPException(status_code=502, detail="Gemini processing failed")

            if attempt % 5 == 0:
                logger.info("[%s] Gemini still pending (attempt %d/%d)", task_id, attempt + 1, max_attempts)

        logger.error("[%s] Gemini task did not complete within %s seconds", task_id, max_poll_seconds)
        raise HTTPException(status_code=504, detail="Gemini processing timed out")


async def call_face_piapi_swap(image_bytes: bytes, face_meta: dict = None, timeout: int = 180) -> bytes:
    """
    Call PIAPI (Image-Toolkit) to do face swapping on a single image.
    Model used: Qubico/image-toolkit
    - Uses follow_redirects=True
    - Retries 5xx transient errors
    - Prefers image_b64 in response for full quality; falls back to output_url
    - Falls back to DZINE_URL if configured; final degraded fallback returns original bytes
    """
    import asyncio

    # small helper to try to parse json and prefer image_b64
    async def _parse_image_response(resp_obj: httpx.Response, client: httpx.AsyncClient):
        try:
            j = resp_obj.json()
        except Exception:
            return resp_obj.content
        if isinstance(j, dict):
            if j.get("image_b64"):
                return base64.b64decode(j["image_b64"])
            if j.get("output_url"):
                url = j.get("output_url")
                if "img.theapi.app/ephemeral" in url:
                    logger.warning("[face-swap] Ephemeral CDN image detected; quality may be reduced. URL: %s", url)
                r_get = await client.get(url)
                r_get.raise_for_status()
                return r_get.content
        return resp_obj.content

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
                logger.exception("[face-swap] Failed to reach PIAPI image-toolkit: %s", e)
                r = None

            # retry transient 5xx up to 2 times
            if r is not None:
                for attempt in range(2):
                    if r.status_code >= 500:
                        logger.warning("[face-swap] PIAPI returned %s, retry %d/2 ...", r.status_code, attempt + 1)
                        await asyncio.sleep(1.5 * (attempt + 1))
                        try:
                            r = await client.post(PIAPI_URL, headers=headers, data={"payload": json.dumps(data)}, files=files)
                        except Exception as e:
                            logger.warning("[face-swap] retry failed: %s", e)
                            r = None
                            break
                    else:
                        break

            if r is None or (r.status_code >= 500):
                # fallback to DZINE if present
                if DZINE_URL:
                    logger.info("[face-swap] Falling back to DZINE_URL for face-swap")
                    try:
                        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as dz_client:
                            dz_files = {"file": ("page.png", image_bytes, "image/png")}
                            dz_data = {"meta": json.dumps(face_meta or {})}
                            dz_r = await dz_client.post(DZINE_URL, headers={}, data=dz_data, files=dz_files)
                            if dz_r.status_code < 400:
                                return await _parse_image_response(dz_r, dz_client)
                    except Exception as e:
                        logger.exception("[face-swap] DZINE fallback failed: %s", e)

                # degraded fallback: return identity (no face swap)
                logger.error("[face-swap] Both PIAPI and fallback failed; returning original image bytes.")
                return image_bytes

            # success path: parse result and prefer image_b64
            try:
                return await _parse_image_response(r, client)
            except Exception as e:
                logger.exception("[face-swap] Failed parsing response, returning raw content: %s", e)
                return r.content

    # fallback to DZINE_URL if configured
    if DZINE_URL:
        headers = {}
        files = {"file": ("page.png", image_bytes, "image/png")}
        data = {"meta": json.dumps(face_meta or {})}
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            r = await client.post(DZINE_URL, headers=headers, data=data, files=files)
            r.raise_for_status()
            try:
                j = r.json()
                if j.get("image_b64"):
                    return base64.b64decode(j["image_b64"])
                if j.get("output_url"):
                    resp = await client.get(j["output_url"])
                    resp.raise_for_status()
                    return resp.content
            except Exception:
                return r.content

    # identity (no face-swap configured)
    return image_bytes

# ---------- endpoints ----------

import httpx, json

@app.get("/test-piapi")
async def test_piapi():
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get("https://api.piapi.ai/api/v1/task")
            return {"status": r.status_code, "text": r.text[:200]}
    except Exception as e:
        return {"error": str(e)}

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

    # create job record
    _jobs[job_id] = {"status": "processing", "created_at": datetime.utcnow().isoformat(), "error": None, "r2_key": None}

    # start background pipeline
    background_tasks.add_task(_process_pipeline, job_id, upload_key, kidinfo, story_choice, gender)

    return {"job_id": job_id, "status": "processing"}

@app.get("/job-status/{job_id}")
async def job_status(job_id: str):
    data = _jobs.get(job_id)
    if not data:
        raise HTTPException(status_code=404, detail="job_id not found")
    if data.get("status") == "completed" and data.get("r2_key"):
        try:
            download_url = generate_presigned_get(CF_R2_BUCKET_OUTPUTS, data["r2_key"])
        except Exception:
            download_url = None
        return {"job_id": job_id, "status": "completed", "download_url": download_url, "r2_key": data.get("r2_key")}
    return {"job_id": job_id, "status": data.get("status", "processing"), "error": data.get("error")}

@app.get("/health")
async def health():
    return {"ok": True, "time": datetime.utcnow().isoformat()}

# ---------- background pipeline (two-phase: text then face) ----------
async def _process_pipeline(job_id: str, kidinfo_r2_key: str, kidinfo: dict, story_choice: str, gender: str):
    """
    Steps:
      A) list template images under templates/{story_choice}/{gender}/ (cap to 5)
         for each: download -> TEXT_PIAPI text-swap
         store text-swapped bytes locally and (optionally) in outputs/tmp/
         AFTER all 5 text-swaps, emit checkpoint log line.
      B) run FACE swap over the 5 text-swapped images
         upload final pages to outputs/{job_id}/NNN.png
      C) assemble final PDF once and upload
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

        tmp_dir = f"/tmp/fictales_{job_id}"
        os.makedirs(tmp_dir, exist_ok=True)

        # ---------- PHASE A: TEXT-SWAP ALL 5 PAGES ----------
        text_swapped_list: List[tuple[str, bytes]] = []
        for idx, tpl_key in enumerate(keys_sorted, start=1):
            logger.info("[%s] [TEXT] downloading template %s", job_id, tpl_key)
            try:
                tpl_bytes = s3_get_bytes(CF_R2_BUCKET_TEMPLATES, tpl_key)
            except Exception as e:
                logger.exception("[%s] failed downloading template %s: %s", job_id, tpl_key, e)
                _jobs[job_id].update(status="failed", error=f"Failed downloading template {tpl_key}")
                return

            try:
                text_swapped_bytes = await call_text_piapi_image_swap(
                    tpl_bytes, kidinfo, template_meta={"template_key": tpl_key, "index": idx}
                )
            except Exception as e:
                logger.exception("[%s] TEXT_PIAPI text-swap failed for %s: %s", job_id, tpl_key, e)
                _jobs[job_id].update(status="failed", error=f"TEXT_PIAPI text-swap failed for {tpl_key}")
                return

            page_num = str(idx).zfill(3)
            # Save interim text-swapped file locally (and optionally to R2 tmp path)
            text_local_path = os.path.join(tmp_dir, f"text_{page_num}.png")
            with open(text_local_path, "wb") as f:
                f.write(text_swapped_bytes)
            text_swapped_list.append((page_num, text_swapped_bytes))

            # Optional: upload interim text-swapped page (helps debugging)
            try:
                tmp_r2_key = f"outputs/{job_id}/_textswap/{page_num}.png"
                s3_put_bytes(CF_R2_BUCKET_OUTPUTS, tmp_r2_key, text_swapped_bytes, content_type="image/png")
                logger.info("[%s] uploaded interim text-swapped -> %s", job_id, tmp_r2_key)
            except Exception as e:
                logger.warning("[%s] failed uploading interim text page (non-fatal): %s", job_id, e)

        # >>>>>>> REQUIRED CHECKPOINT LOG <<<<<<<
        logger.info('done with all 5 page text swaps, moving on to face swaps now, pls wait')

        # ---------- PHASE B: FACE-SWAP OVER TEXT-SWAPPED RESULTS ----------
        final_image_paths: List[str] = []
        for page_num, text_swapped_bytes in text_swapped_list:
            idx = int(page_num)
            try:
                face_swapped_bytes = await call_face_piapi_swap(
                    text_swapped_bytes,
                    face_meta={"kidinfo_key": kidinfo_r2_key, "index": idx}
                )
            except Exception as e:
                logger.exception("[%s] face-swap failed for page %s: %s", job_id, page_num, e)
                _jobs[job_id].update(status="failed", error=f"Face-swap failed for page {page_num}")
                return

            local_path = os.path.join(tmp_dir, f"{page_num}.png")
            with open(local_path, "wb") as f:
                f.write(face_swapped_bytes)
            final_image_paths.append(local_path)

            r2_out_key = f"outputs/{job_id}/{page_num}.png"
            try:
                s3_put_bytes(CF_R2_BUCKET_OUTPUTS, r2_out_key, face_swapped_bytes, content_type="image/png")
                logger.info("[%s] uploaded final page -> %s", job_id, r2_out_key)
            except Exception as e:
                logger.exception("[%s] failed uploading page to outputs: %s", job_id, e)
                _jobs[job_id].update(status="failed", error="Failed uploading page image to outputs")
                return

        # ---------- PHASE C: ASSEMBLE PDF ONCE ----------
        pdf_path = os.path.join(tmp_dir, f"{job_id}.pdf")
        try:
            images = []
            for p in final_image_paths:
                img = Image.open(p)
                if img.mode in ("RGBA", "LA") or (img.mode == "P"):
                    img = img.convert("RGB")
                images.append(img)
            if not images:
                raise RuntimeError("No images to assemble")
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
    uvicorn.run("app.main:app", host="0.0.0.0", port=port)

