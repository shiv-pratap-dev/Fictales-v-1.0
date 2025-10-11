# app/main.py
"""
Fictales Railway orchestrator

Endpoints:
- POST /get-presigned-upload   -> returns presigned PUT URL + r2_key for frontend upload
- POST /generate-story        -> enqueue background job that:
     1) calls HF worker stage=hybrid
     2) calls HF worker stage=faceswap (passes templates_bucket)
     3) collects result (r2_key or inline base64 PDF), uploads to outputs if needed
- GET  /job-status/{job_id}   -> job state + presigned GET if completed
- GET  /health                -> basic healthcheck

ENV expected (set these in Railway project variables):
- CF_ACCOUNT_ID
- CF_R2_BUCKET_UPLOADS (default fictales-uploads)
- CF_R2_BUCKET_OUTPUTS (default fictales-outputs)
- CF_R2_BUCKET_TEMPLATES (optional; default fictales-templates)
- CF_ACCESS_KEY_ID  OR CF_PRESIGN_ACCESS_KEY  (presigner credentials)
- CF_SECRET_ACCESS_KEY OR CF_PRESIGN_SECRET_KEY
- HF_SPACE_URL  (e.g. https://shivlyaa-fictales-worker.hf.space)
- HF_API_TOKEN  (optional; for private HF spaces)
- PRESIGN_EXPIRATION (seconds; optional; default 3600)
"""

import os
import uuid
import logging
import base64
from datetime import datetime
from typing import Optional, Dict, Any, Tuple
import boto3
import httpx
from fastapi import FastAPI, BackgroundTasks, HTTPException, UploadFile, Form, File
from pydantic import BaseModel
from botocore.client import Config
from typing import Literal, Optional

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ficta-railway")

# ---------- Environment / Config ----------
CF_ACCOUNT_ID = os.getenv("CF_ACCOUNT_ID")
CF_R2_BUCKET_UPLOADS = os.getenv("CF_R2_BUCKET_UPLOADS", "fictales-uploads")
CF_R2_BUCKET_OUTPUTS = os.getenv("CF_R2_BUCKET_OUTPUTS", "fictales-outputs")
CF_R2_BUCKET_TEMPLATES = os.getenv("CF_R2_BUCKET_TEMPLATES", "fictales-templates")

# Support a couple of env-name variants (use presigner creds if provided)
CF_ACCESS_KEY_ID = os.getenv("CF_ACCESS_KEY_ID") or os.getenv("CF_PRESIGN_ACCESS_KEY") or os.getenv("CF_PRESIGN_AK")
CF_SECRET_ACCESS_KEY = os.getenv("CF_SECRET_ACCESS_KEY") or os.getenv("CF_PRESIGN_SECRET_KEY") or os.getenv("CF_PRESIGN_SK")

HF_SPACE_URL = os.getenv("HF_SPACE_URL")  # must be full URL, e.g. https://...-hf.space
HF_API_TOKEN = os.getenv("HF_API_TOKEN")  # optional

PRESIGN_EXPIRATION = int(os.getenv("PRESIGN_EXPIRATION", "3600"))

# Quick sanity logging of env presence
if not CF_ACCOUNT_ID:
    logger.warning("CF_ACCOUNT_ID missing; s3 endpoint won't be built.")

if not (CF_ACCESS_KEY_ID and CF_SECRET_ACCESS_KEY):
    logger.warning("Cloudflare R2 credentials missing (CF_ACCESS_KEY_ID / CF_SECRET_ACCESS_KEY). Presign & S3 ops will fail.")

if not HF_SPACE_URL:
    logger.warning("HF_SPACE_URL missing; HF calls will fail until provided.")

# ---------- S3 / Cloudflare R2 client ----------
R2_ENDPOINT = f"https://{CF_ACCOUNT_ID}.r2.cloudflarestorage.com" if CF_ACCOUNT_ID else None

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
        logger.exception("Failed to initialize S3 client: %s", e)
        s3 = None

# ---------- FastAPI app ----------
app = FastAPI(title="FictalesRailwayCore")

# ---------- in-memory job store (simple) ----------
_jobs: Dict[str, Dict[str, Any]] = {}

# ---------- request / response models ----------
class PresignRequest(BaseModel):
    filename: str
    content_type: Optional[str] = None

class PresignResp(BaseModel):
    upload_url: str
    r2_key: str
    bucket: str
    expires_in: int

class UploadRef(BaseModel):
    r2_key: str
    bucket: Optional[str] = None

class GenerateRequest(BaseModel):
    job_id: Optional[str] = None
    upload_ref: UploadRef
    kid_name: str
    story_choice: str
    gender: str
    callback_url: Optional[str] = None
    callback_secret: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None

class JobStatusResp(BaseModel):
    job_id: str
    status: str
    download_url: Optional[str] = None
    r2_key: Optional[str] = None
    error: Optional[str] = None

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
    url = s3.generate_presigned_url(ClientMethod="put_object", Params=params, ExpiresIn=expires_in)
    return url

def generate_presigned_get(bucket: str, key: str, expires_in: int = PRESIGN_EXPIRATION) -> str:
    if s3 is None:
        raise RuntimeError("S3 client not configured")
    params = {"Bucket": bucket, "Key": key}
    url = s3.generate_presigned_url(ClientMethod="get_object", Params=params, ExpiresIn=expires_in)
    return url

def s3_object_exists(bucket: str, key: str) -> bool:
    if s3 is None:
        return False
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False

async def call_hf_process(payload: dict, endpoint: str = "/process", timeout: int = 120) -> dict:
    if not HF_SPACE_URL:
        raise RuntimeError("HF_SPACE_URL not configured")
    url = HF_SPACE_URL.rstrip("/") + endpoint
    headers = {"Content-Type": "application/json"}
    if HF_API_TOKEN:
        headers["Authorization"] = f"Bearer {HF_API_TOKEN}"
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(url, json=payload, headers=headers)
        r.raise_for_status()
        try:
            return r.json()
        except Exception:
            return {"raw_text": r.text}

def extract_r2_key_from_hf_result(result: Any) -> Optional[Any]:
    """
    Interpret HF worker result:
    - If dict contains 'r2_key' return string
    - If dict contains 'pdf_bytes' or 'pdf_b64' return ("__pdf_base64__", b64str)
    - Else return None
    """
    if not result:
        return None
    if isinstance(result, dict):
        if result.get("r2_key"):
            return result["r2_key"]
        if result.get("r2_key_full"):
            return result["r2_key_full"]
        if result.get("pdf_bytes"):
            return ("__pdf_base64__", result["pdf_bytes"])
        if result.get("pdf_b64"):
            return ("__pdf_base64__", result["pdf_b64"])
        for v in result.values():
            if isinstance(v, str) and v.startswith("outputs/"):
                return v
    if isinstance(result, str) and result.startswith("outputs/"):
        return result
    return None

# ---------- API endpoints ----------
@app.post("/get-presigned-upload", response_model=PresignResp)
async def get_presigned_upload(req: PresignRequest):
    if s3 is None:
        raise HTTPException(status_code=500, detail="R2 not configured on server")
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
    file: UploadFile = File(...),
    kid_name: str = Form(...),
    story_choice: Literal["story1", "story2"] = Form(...),
    gender: Literal["boy", "girl"] = Form(...),
    callback_url: Optional[str] = Form(None),
    callback_secret: Optional[str] = Form(None),
):
    """
    Single-step endpoint for uploads + job creation:
    - Accepts file via Swagger / frontend form
    - Uploads file to Cloudflare R2 (uploads bucket)
    - Creates a job and starts background HF orchestration (_process_pipeline)
    """
    # Config check
    if not (CF_ACCOUNT_ID and CF_ACCESS_KEY_ID and CF_SECRET_ACCESS_KEY and HF_SPACE_URL):
        raise HTTPException(status_code=500, detail="Server not fully configured (missing env vars)")

    job_id = uuid.uuid4().hex
    logger.info("Received file-upload job=%s name=%s story=%s gender=%s", job_id, kid_name, story_choice, gender)

    # Read uploaded bytes
    try:
        content = await file.read()
    except Exception as e:
        logger.exception("[%s] failed reading uploaded file: %s", job_id, e)
        raise HTTPException(status_code=400, detail="Failed to read uploaded file")

    # Prepare R2 key and upload kwargs
    upload_key = f"uploads/{job_id}_{file.filename}"
    if s3 is None:
        logger.error("[%s] S3/R2 client not configured", job_id)
        raise HTTPException(status_code=500, detail="S3/R2 client not configured on server")

    put_kwargs = {"Bucket": CF_R2_BUCKET_UPLOADS, "Key": upload_key, "Body": content}
    if file.content_type:
        put_kwargs["ContentType"] = file.content_type

    # Upload to R2
    try:
        s3.put_object(**put_kwargs)
        logger.info("[%s] Uploaded input file to R2: %s", job_id, upload_key)
    except Exception as e:
        logger.exception("[%s] Failed uploading to R2: %s", job_id, e)
        raise HTTPException(status_code=500, detail="Failed to upload input file to R2")

    # Initialize job record
    _jobs[job_id] = {
        "status": "processing",
        "created_at": datetime.utcnow().isoformat(),
        "error": None,
        "r2_key": None,
    }

    # Build the request dict expected by the pipeline
    req_dict = {
        "job_id": job_id,
        "upload_ref": {"r2_key": upload_key, "bucket": CF_R2_BUCKET_UPLOADS},
        "kid_name": kid_name,
        "story_choice": story_choice,
        "gender": gender,
        "callback_url": callback_url,
        "callback_secret": callback_secret,
    }

    # Kick off background orchestration
    background_tasks.add_task(_process_pipeline, job_id, req_dict)

    return {"job_id": job_id, "status": "processing"}
@app.get("/job-status/{job_id}", response_model=JobStatusResp)
async def job_status(job_id: str):
    data = _jobs.get(job_id)
    if not data:
        raise HTTPException(status_code=404, detail="job_id not found")
    if data.get("status") == "completed" and data.get("r2_key"):
        try:
            download_url = generate_presigned_get(CF_R2_BUCKET_OUTPUTS, data["r2_key"])
        except Exception:
            download_url = None
        return JobStatusResp(job_id=job_id, status="completed", download_url=download_url, r2_key=data.get("r2_key"))
    return JobStatusResp(job_id=job_id, status=data.get("status", "processing"), error=data.get("error"))

@app.get("/health")
async def health():
    return {"ok": True, "time": datetime.utcnow().isoformat()}

# ---------- background orchestration ----------
async def _process_pipeline(job_id: str, req: dict):
    """
    Orchestrate:
      1) HF hybrid stage
      2) HF faceswap stage (templates bucket provided)
      3) collect final result (r2_key or inline PDF) and store in outputs
    """
    try:
        upload_ref = req.get("upload_ref") or {}
        upload_bucket = upload_ref.get("bucket") or CF_R2_BUCKET_UPLOADS
        upload_key = upload_ref.get("r2_key")

        if not upload_key:
            msg = "No upload r2_key provided"
            logger.error("[%s] %s", job_id, msg)
            _jobs[job_id].update(status="failed", error=msg)
            return

        # sanity check uploaded object exists
        if not s3_object_exists(upload_bucket, upload_key):
            msg = f"Uploaded file not found: {upload_bucket}/{upload_key}"
            logger.error("[%s] %s", job_id, msg)
            _jobs[job_id].update(status="failed", error=msg)
            return

        base_payload = {
            "job_id": job_id,
            "uploads": [{"bucket": upload_bucket, "r2_key": upload_key}],
            "settings": {
                "kid_name": req.get("kid_name"),
                "story_choice": req.get("story_choice"),
                "gender": req.get("gender"),
                **(req.get("extra") or {}),
            },
        }

        # ---------- 1) hybrid step ----------
        logger.info("[%s] calling HF hybrid step", job_id)
        hybrid_payload = {**base_payload, "stage": "hybrid"}
        try:
            hybrid_result = await call_hf_process(hybrid_payload, endpoint="/process", timeout=180)
            logger.info("[%s] hybrid_result (truncated): %s", job_id, str(hybrid_result)[:500])
        except Exception as e:
            msg = f"HF hybrid step failed: {e}"
            logger.exception("[%s] %s", job_id, msg)
            _jobs[job_id].update(status="failed", error=msg)
            return

        hybrid_ref = extract_r2_key_from_hf_result(hybrid_result)

        # ---------- 2) faceswap step ----------
        faceswap_payload = {**base_payload, "stage": "faceswap", "templates_bucket": CF_R2_BUCKET_TEMPLATES}
        if hybrid_ref:
            faceswap_payload["hybrid_ref"] = hybrid_ref

        logger.info("[%s] calling HF faceswap step", job_id)
        try:
            faceswap_result = await call_hf_process(faceswap_payload, endpoint="/process", timeout=600)
            logger.info("[%s] faceswap_result (truncated): %s", job_id, str(faceswap_result)[:500])
        except Exception as e:
            msg = f"HF faceswap step failed: {e}"
            logger.exception("[%s] %s", job_id, msg)
            _jobs[job_id].update(status="failed", error=msg)
            return

        # ---------- 3) interpret final result ----------
        final = extract_r2_key_from_hf_result(faceswap_result)
        if isinstance(final, str):
            final_r2_key = final
            # verify existence
            if not s3_object_exists(CF_R2_BUCKET_OUTPUTS, final_r2_key):
                # attempt tolerant check if worker returned "outputs/<name>"
                if final_r2_key.startswith("outputs/") and s3_object_exists(CF_R2_BUCKET_OUTPUTS, final_r2_key):
                    pass
                else:
                    msg = f"HF reported output {final_r2_key} but not found in R2"
                    logger.error("[%s] %s", job_id, msg)
                    _jobs[job_id].update(status="failed", error=msg)
                    return
            _jobs[job_id].update(status="completed", r2_key=final_r2_key)
            logger.info("[%s] pipeline completed => %s", job_id, final_r2_key)
            return

        elif isinstance(final, tuple) and final[0] == "__pdf_base64__":
            b64 = final[1]
            try:
                pdf_bytes = base64.b64decode(b64)
            except Exception as e:
                logger.exception("[%s] failed decoding base64 PDF: %s", job_id, e)
                _jobs[job_id].update(status="failed", error="Invalid base64 from worker")
                return
            pdf_key = f"outputs/{job_id}.pdf"
            try:
                s3.put_object(Bucket=CF_R2_BUCKET_OUTPUTS, Key=pdf_key, Body=pdf_bytes, ContentType="application/pdf")
                _jobs[job_id].update(status="completed", r2_key=pdf_key)
                logger.info("[%s] uploaded inline PDF => %s", job_id, pdf_key)
                return
            except Exception as e:
                logger.exception("[%s] failed uploading inline PDF: %s", job_id, e)
                _jobs[job_id].update(status="failed", error="Failed uploading PDF to R2")
                return

        else:
            msg = "HF worker did not return recognizable output (r2_key or pdf_b64)."
            logger.error("[%s] %s result=%s", job_id, msg, str(faceswap_result)[:500])
            _jobs[job_id].update(status="failed", error=msg)
            return

    except Exception as exc:
        logger.exception("[%s] unexpected error in pipeline: %s", job_id, exc)
        _jobs[job_id].update(status="failed", error=str(exc))
        return

# ---------- runner fallback ----------
if __name__ == "__main__":
    import uvicorn, os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port)

