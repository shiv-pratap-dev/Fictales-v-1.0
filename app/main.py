# app/main.py
"""
Fictales Railway core (updated env var names to avoid collisions)

Key env changes:
- Face-swap (existing, keep as-is):   PIAPI_URL, PIAPI_KEY
- Text-swap (new separate account):   TEXT_PIAPI_URL, TEXT_PIAPI_KEY
- Optional old face-swap service (fallback): DZINE_URL (kept as optional)

Other ENV expected (same as before):
- CF_ACCOUNT_ID
- CF_R2_BUCKET_UPLOADS (default: fictales-uploads)
- CF_R2_BUCKET_OUTPUTS (default: fictales-outputs)
- CF_R2_BUCKET_TEMPLATES (default: fictales-templates)
- CF_ACCESS_KEY_ID / CF_SECRET_ACCESS_KEY
- PRESIGN_EXPIRATION (optional)
"""
import os
import uuid
import json
import logging
import base64
from datetime import datetime
from typing import Optional, Dict, Any, List
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

# FACE-SWAP service (keep these exact names)
PIAPI_URL = os.getenv("PIAPI_URL")       # <-- Face-swap PiAPI (existing account)
PIAPI_KEY = os.getenv("PIAPI_KEY")

# TEXT-SWAP service (NEW names to avoid collision)
TEXT_PIAPI_URL = os.getenv("TEXT_PIAPI_URL")   # <-- Text-swap PiAPI (separate account)
TEXT_PIAPI_KEY = os.getenv("TEXT_PIAPI_KEY")

# Optional legacy / alternative face-swap endpoint (kept as fallback; if set, may be used instead)
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
app = FastAPI(title="FictalesRailwayCore (Separated PiAPI envs)")

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
    # sort by final filename numeric parts if possible; fallback to lexicographic
    base = os.path.basename(name)
    parts = base.split(".")[0].split("_")
    for p in reversed(parts):
        if p.isdigit():
            return int(p)
    # fallback
    return base

# ---------- external API helpers ----------
async def call_text_piapi_image_swap(image_bytes: bytes, kidinfo: dict, template_meta: dict = None, timeout: int = 120) -> bytes:
    """
    Call TEXT_PIAPI (separate account) to do text swap on a single image.
    Uses TEXT_PIAPI_URL and TEXT_PIAPI_KEY.
    Contract assumed: returns JSON with "image_b64" or direct bytes.
    """
    if not TEXT_PIAPI_URL:
        raise RuntimeError("TEXT_PIAPI_URL not configured")

    headers = {"Authorization": f"Bearer {TEXT_PIAPI_KEY}"} if TEXT_PIAPI_KEY else {}
    data = {"kidinfo": json.dumps(kidinfo)}
    if template_meta:
        data["template_meta"] = json.dumps(template_meta)

    files = {"file": ("page.png", image_bytes, "image/png")}
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(TEXT_PIAPI_URL, headers=headers, data=data, files=files)
        r.raise_for_status()
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

async def call_face_piapi_swap(image_bytes: bytes, face_meta: dict = None, timeout: int = 180) -> bytes:
    """
    Call PIAPI (face-swap account) to do face swapping on a single image.
    Uses PIAPI_URL and PIAPI_KEY.
    If PIAPI_URL is not set but DZINE_URL is present, it will call DZINE_URL instead.
    If neither present, returns identity (original bytes).
    """
    # Prefer PIAPI (face-swap account)
    if PIAPI_URL:
        headers = {"Authorization": f"Bearer {PIAPI_KEY}"} if PIAPI_KEY else {}
        data = {"meta": json.dumps(face_meta or {})}
        files = {"file": ("page.png", image_bytes, "image/png")}
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(PIAPI_URL, headers=headers, data=data, files=files)
            r.raise_for_status()
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

    # fallback to DZINE if provided
    if DZINE_URL:
        headers = {}
        files = {"file": ("page.png", image_bytes, "image/png")}
        data = {"meta": json.dumps(face_meta or {})}
        async with httpx.AsyncClient(timeout=timeout) as client:
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
                pass
            return r.content

    # identity (no face-swap configured)
    return image_bytes

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
    # either user uploaded kidinfo earlier and passes upload_ref, or provides kidinfo JSON in form
    upload_ref: Optional[str] = Form(None),  # r2_key of kidinfo JSON if pre-uploaded
    kidinfo_json: Optional[str] = Form(None),  # JSON string for kidinfo (if no upload_ref)
    story_choice: str = Form(...),  # e.g. "story1" or "story2"
    gender: str = Form(...),
):
    if s3 is None:
        raise HTTPException(status_code=500, detail="Server storage not configured")

    job_id = uuid.uuid4().hex
    logger.info("Creating job %s for story=%s gender=%s", job_id, story_choice, gender)

    kidinfo = None
    kidinfo_r2_key = None

    if upload_ref:
        kidinfo_r2_key = upload_ref
        if not s3_object_exists(CF_R2_BUCKET_UPLOADS, kidinfo_r2_key):
            raise HTTPException(status_code=400, detail="Provided upload_ref not found in uploads bucket")
        try:
            raw = s3_get_bytes(CF_R2_BUCKET_UPLOADS, kidinfo_r2_key)
            kidinfo = json.loads(raw.decode("utf-8"))
        except Exception as e:
            logger.exception("Failed reading kidinfo from R2: %s", e)
            raise HTTPException(status_code=500, detail="Failed to read kidinfo from R2")
    elif kidinfo_json:
        try:
            kidinfo = json.loads(kidinfo_json)
        except Exception as e:
            raise HTTPException(status_code=400, detail="kidinfo_json invalid JSON")
        kidinfo_r2_key = f"uploads/kidinfo_{job_id}.json"
        try:
            s3_put_bytes(CF_R2_BUCKET_UPLOADS, kidinfo_r2_key, json.dumps(kidinfo).encode("utf-8"), content_type="application/json")
            logger.info("[%s] uploaded kidinfo -> %s", job_id, kidinfo_r2_key)
        except Exception as e:
            logger.exception("[%s] failed uploading kidinfo: %s", job_id, e)
            raise HTTPException(status_code=500, detail="Failed to upload kidinfo to R2")
    else:
        raise HTTPException(status_code=400, detail="Either upload_ref or kidinfo_json must be provided")

    _jobs[job_id] = {"status": "processing", "created_at": datetime.utcnow().isoformat(), "error": None, "r2_key": None}
    background_tasks.add_task(_process_pipeline, job_id, kidinfo_r2_key, kidinfo, story_choice, gender)
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

# ---------- background pipeline ----------
async def _process_pipeline(job_id: str, kidinfo_r2_key: str, kidinfo: dict, story_choice: str, gender: str):
    """
    Steps:
      1) list template images under templates/{story_choice}/
      2) for each template image: download -> call TEXT_PIAPI for text swap -> save swapped image
      3) call PIAPI (face-swap account) per image -> save final images
      4) assemble final PDF from final images and upload to outputs/{job_id}/{job_id}.pdf
    """
    try:
        prefix = f"{story_choice}/"
        logger.info("[%s] listing templates at %s/%s", job_id, CF_R2_BUCKET_TEMPLATES, prefix)
        try:
            keys = s3_list_keys(CF_R2_BUCKET_TEMPLATES, prefix)
        except Exception as e:
            logger.exception("[%s] failed listing templates: %s", job_id, e)
            _jobs[job_id].update(status="failed", error="Failed listing template files")
            return

        if not keys:
            msg = f"No templates found for {story_choice} in templates bucket"
            logger.error("[%s] %s", job_id, msg)
            _jobs[job_id].update(status="failed", error=msg)
            return

        keys_sorted = sorted(keys, key=lambda k: natural_sort_key(k))
        tmp_dir = f"/tmp/fictales_{job_id}"
        os.makedirs(tmp_dir, exist_ok=True)
        final_image_paths = []

        for idx, tpl_key in enumerate(keys_sorted, start=1):
            logger.info("[%s] processing template %s", job_id, tpl_key)
            try:
                tpl_bytes = s3_get_bytes(CF_R2_BUCKET_TEMPLATES, tpl_key)
            except Exception as e:
                logger.exception("[%s] failed downloading template %s: %s", job_id, tpl_key, e)
                _jobs[job_id].update(status="failed", error=f"Failed downloading template {tpl_key}")
                return

            # 1) TEXT swap using TEXT_PIAPI (separate account)
            try:
                text_swapped_bytes = await call_text_piapi_image_swap(tpl_bytes, kidinfo, template_meta={"template_key": tpl_key, "index": idx})
            except Exception as e:
                logger.exception("[%s] TEXT_PIAPI text-swap failed for %s: %s", job_id, tpl_key, e)
                _jobs[job_id].update(status="failed", error=f"TEXT_PIAPI text-swap failed for {tpl_key}")
                return

            # 2) FACE swap using PIAPI (face-swap account) or DZINE fallback
            try:
                face_swapped_bytes = await call_face_piapi_swap(text_swapped_bytes, face_meta={"kidinfo_key": kidinfo_r2_key, "index": idx})
            except Exception as e:
                logger.exception("[%s] face-swap failed for page %s: %s", job_id, idx, e)
                _jobs[job_id].update(status="failed", error=f"Face-swap failed for page {idx}")
                return

            page_num = str(idx).zfill(3)
            local_path = os.path.join(tmp_dir, f"{page_num}.png")
            with open(local_path, "wb") as f:
                f.write(face_swapped_bytes)
            final_image_paths.append(local_path)

            r2_out_key = f"outputs/{job_id}/{page_num}.png"
            try:
                s3_put_bytes(CF_R2_BUCKET_OUTPUTS, r2_out_key, face_swapped_bytes, content_type="image/png")
                logger.info("[%s] uploaded page -> %s", job_id, r2_out_key)
            except Exception as e:
                logger.exception("[%s] failed uploading page to outputs: %s", job_id, e)
                _jobs[job_id].update(status="failed", error="Failed uploading page image to outputs")
                return

        # assemble PDF once
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
