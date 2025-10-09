# app/main.py
import os
import re
import uuid
import shutil
import logging
import base64
import asyncio
import json
from pathlib import Path
from typing import Optional, Any, List
from enum import Enum

import aiofiles
import httpx
import numpy as np
import easyocr
import boto3
from botocore.client import Config
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont

# -------- CONFIG ----------
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
ALLOWED_EXT = {".png", ".jpg", ".jpeg"}

# Base directories (project root -> storage)
BASE_DIR = Path(__file__).resolve().parent.parent

# Load .env (optional)
load_dotenv(dotenv_path=BASE_DIR / ".env")

# PiAPI defaults (exact values from your old main.py)
PIAPI_URL = os.getenv("PIAPI_URL", "https://api.piapi.ai/api/v1/task")
PIAPI_KEY = os.getenv("PIAPI_KEY")  # set this in .env to enable real calls
PIAPI_MODEL = os.getenv("PIAPI_MODEL", "Qubico/image-toolkit")

PIAPI_POLL_INTERVAL = float(os.getenv("PIAPI_POLL_INTERVAL", "2.0"))
PIAPI_POLL_MAX = int(os.getenv("PIAPI_POLL_MAX", "20"))

# Cloudflare R2 / S3 config (optional)
R2_ENDPOINT = os.getenv("R2_ENDPOINT")  # e.g. https://<account_id>.r2.cloudflarestorage.com
R2_ACCESS_KEY = os.getenv("R2_ACCESS_KEY")
R2_SECRET_KEY = os.getenv("R2_SECRET_KEY")
R2_BUCKET = os.getenv("R2_BUCKET")

# Paths matching your project layout (from screenshot)
STORAGE_DIR = BASE_DIR / "storage"
PAGES_DIR = STORAGE_DIR / "pages"                       # legacy single page: storage/pages/story.png
STORIES_DIR = STORAGE_DIR / "stories"                   # storage/stories/<story>/<gender>/*.png
INPUT_DIR = STORAGE_DIR / "input" / "kid_faces"         # uploaded faces stored here
OUTPUT_DIR = STORAGE_DIR / "output" / "swapped_pages"   # per-page swapped outputs
PDF_DIR = STORAGE_DIR / "output" / "pdfs"               # assembled PDFs

ASSETS_URL_PREFIX = "/assets"  # mounted static URL prefix

# -------- LOGGING ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fictales")

# -------- APP ----------
app = FastAPI(title="Fictales - Face Swap + Hybrid3 Name Swap")

# --- CORS fix ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount(ASSETS_URL_PREFIX, StaticFiles(directory=str(STORAGE_DIR)), name="assets")

# Ensure directories exist
@app.on_event("startup")
async def ensure_dirs():
    for p in (PAGES_DIR, STORIES_DIR, INPUT_DIR, OUTPUT_DIR, PDF_DIR):
        os.makedirs(p, exist_ok=True)
    os.makedirs(STORAGE_DIR / "tmp", exist_ok=True)
    logger.info("Storage directories checked/created.")

# ---------- ENUMS for Swagger choices ----------
class Gender(str, Enum):
    boy = "boy"
    girl = "girl"

class StoryChoice(str, Enum):
    story1 = "story1"
    story2 = "story2"

# ---------- Utilities ----------
def _ext_from_filename(filename: str) -> str:
    return Path(filename).suffix.lower()

async def save_bytes_to_path(content_bytes: bytes, dest_path: Path) -> None:
    async with aiofiles.open(dest_path, "wb") as f:
        await f.write(content_bytes)

def file_bytes_to_b64_str(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")

def natural_sort_key(s: str):
    """Natural sort key so page2.png comes before page10.png."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def _find_image_in_output(output_obj: Any) -> Optional[tuple]:
    """
    Search for an image URL or base64 image inside an arbitrary API response object.
    Returns ('url', url_str) or ('b64', b64_str) or None.
    """
    if output_obj is None:
        return None
    if isinstance(output_obj, str):
        s = output_obj.strip()
        if s.startswith("http"):
            return ("url", s)
        if all(c.isalnum() or c in "+/=\n\r" for c in s) and len(s) > 200:
            return ("b64", s)
        return None
    if isinstance(output_obj, dict):
        for v in output_obj.values():
            found = _find_image_in_output(v)
            if found:
                return found
    if isinstance(output_obj, list):
        for item in output_obj:
            found = _find_image_in_output(item)
            if found:
                return found
    return None

# ---------- R2 / S3 helpers ----------
def get_r2_client():
    if not (R2_ENDPOINT and R2_ACCESS_KEY and R2_SECRET_KEY and R2_BUCKET):
        return None
    return boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT,
        aws_access_key_id=R2_ACCESS_KEY,
        aws_secret_access_key=R2_SECRET_KEY,
        config=Config(signature_version="s3v4"),
        region_name="auto",
    )

def upload_to_r2(local_path: Path, remote_key: str) -> Optional[str]:
    """
    Upload local_path to R2 under remote_key and return public URL.
    Returns None on failure or if R2 not configured.
    """
    client = get_r2_client()
    if client is None:
        return None
    try:
        client.upload_file(str(local_path), R2_BUCKET, remote_key)
        # Public URL (S3 style) - Cloudflare will serve this via their domain (or you can set a custom CDN)
        url = f"{R2_ENDPOINT}/{R2_BUCKET}/{remote_key}"
        return url
    except Exception as e:
        logger.exception("Failed to upload %s to R2: %s", local_path, e)
        return None

# ---------- PiAPI face-swap (unchanged) ----------
async def create_piapi_task_and_wait(target_path: Path, swap_path: Path, output_path: Path) -> Optional[str]:
    """
    Call PiAPI (or compatible endpoint) to create a face-swap task, poll until completion,
    save the resulting image to output_path and return str(output_path). If PIAPI_KEY missing
    or any failure occurs, return None to indicate caller should fallback.
    """
    if not PIAPI_URL or not PIAPI_KEY:
        logger.info("PIAPI not configured (PIAPI_KEY missing). Skipping remote call.")
        return None

    payload = {
        "model": PIAPI_MODEL,
        "task_type": "face-swap",
        "input": {
            "target_image": file_bytes_to_b64_str(target_path),
            "swap_image": file_bytes_to_b64_str(swap_path),
        },
    }

    headers = {"x-api-key": PIAPI_KEY, "Content-Type": "application/json"}

    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            logger.info("Creating PiAPI task...")
            resp = await client.post(PIAPI_URL, json=payload, headers=headers)
            resp.raise_for_status()
            resp_json = resp.json()
            task_id = resp_json.get("data", {}).get("task_id") or resp_json.get("task_id")
            if not task_id:
                logger.error("No task_id returned from create-task response: %s", resp_json)
                return None

            poll_url = f"{PIAPI_URL.rstrip('/')}/{task_id}"
            logger.info("Polling task status at %s", poll_url)

            for attempt in range(PIAPI_POLL_MAX):
                poll_resp = await client.get(poll_url, headers=headers, timeout=30.0)
                if 400 <= poll_resp.status_code < 500:
                    logger.error("Client error while polling: %s", poll_resp.text)
                    poll_resp.raise_for_status()
                poll_json = poll_resp.json()
                status = (poll_json.get("data", {}).get("status") or "").lower()
                logger.info("Poll attempt %d status=%s", attempt + 1, status)

                if status in ("completed", "done", "success", "finished", "complete"):
                    output_obj = poll_json.get("data", {}).get("output")
                    found = _find_image_in_output(output_obj)
                    if found:
                        typ, val = found
                        if typ == "url":
                            logger.info("Downloading image from returned URL...")
                            get_resp = await client.get(val, timeout=60.0)
                            get_resp.raise_for_status()
                            await save_bytes_to_path(get_resp.content, output_path)
                            return str(output_path)
                        else:  # b64
                            logger.info("Decoding base64 image returned by API...")
                            img_bytes = base64.b64decode(val)
                            await save_bytes_to_path(img_bytes, output_path)
                            return str(output_path)
                    # deeper search
                    deeper = _find_image_in_output(poll_json.get("data"))
                    if deeper:
                        typ, val = deeper
                        if typ == "url":
                            get_resp = await client.get(val, timeout=60.0)
                            get_resp.raise_for_status()
                            await save_bytes_to_path(get_resp.content, output_path)
                            return str(output_path)
                        else:
                            img_bytes = base64.b64decode(val)
                            await save_bytes_to_path(img_bytes, output_path)
                            return str(output_path)
                    logger.warning("Task finished but no image found in response: %s", poll_json)
                    return None

                if status in ("failed", "error"):
                    logger.error("Task failed: %s", poll_json.get("data", {}).get("error") or poll_json)
                    return None

                await asyncio.sleep(PIAPI_POLL_INTERVAL)

            logger.error("Timed out polling PiAPI task after %d attempts.", PIAPI_POLL_MAX)
            return None

        except Exception as e:
            logger.exception("Exception while calling PiAPI: %s", e)
            return None

# ---------- Hybrid-3: OCR + erase + redraw helpers ----------
# Create OCR reader (easyocr). Keep GPU disabled.
try:
    OCR_READER = easyocr.Reader(['en'], gpu=False)
except Exception as e:
    logger.warning("EasyOCR initialization failed. OCR will be disabled. Exception: %s", e)
    OCR_READER = None

def load_story_meta(story_dir: Path) -> Optional[dict]:
    meta_path = story_dir / "meta.json"
    if meta_path.exists():
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None

def choose_best_text_candidate(ocr_results: List[Any], new_name: str):
    """
    Pick the best OCR-detected text bbox likely containing a name to replace.
    Heuristics: confidence, length, closeness to bottom/center.
    """
    best = None
    for bbox, text, conf in ocr_results:
        txt = (text or "").strip()
        if not txt:
            continue
        if len(txt) > 25:
            continue
        # simple score
        score = float(conf)
        # prefer similar length to new_name (weak)
        len_diff = abs(len(txt) - len(new_name))
        score = score - (len_diff * 0.01)
        if best is None or score > best[0]:
            best = (score, bbox, txt, conf)
    return best  # returns tuple or None

def bbox_to_xy(bbox):
    # bbox format from easyocr: list of 4 points [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    xs = [int(p[0]) for p in bbox]
    ys = [int(p[1]) for p in bbox]
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))

def sample_background_color(pil_img: Image.Image, bbox, pad=6):
    x1,y1,x2,y2 = bbox
    w,h = pil_img.size
    sx1 = max(0, x1 - pad); sy1 = max(0, y1 - pad)
    sx2 = min(w, x2 + pad); sy2 = min(h, y2 + pad)
    crop = pil_img.crop((sx1, sy1, sx2, sy2))
    arr = np.array(crop).reshape(-1,3)
    # median color
    if arr.size == 0:
        return (255,255,255)
    median = tuple(int(np.median(arr[:,i])) for i in range(3))
    return median

def erase_bbox(pil_img: Image.Image, bbox):
    # Fill bbox with sampled background color (simple and robust)
    x1,y1,x2,y2 = bbox
    color = sample_background_color(pil_img, (x1,y1,x2,y2), pad=8)
    draw = ImageDraw.Draw(pil_img)
    draw.rectangle([x1, y1, x2, y2], fill=color)
    return pil_img

def draw_name(pil_img: Image.Image, bbox, new_name: str, font_path: Optional[str]=None, font_size: Optional[int]=None, color=(0,0,0)):
    x1,y1,x2,y2 = bbox
    draw = ImageDraw.Draw(pil_img)
    h_box = max(1, y2 - y1)
    # approximate font size if not given
    if font_size is None:
        font_size = int(h_box * 0.85)
    # try to load provided font otherwise fallback to a bundled font
    font = None
    if font_path:
        try:
            font = ImageFont.truetype(str(font_path), font_size)
        except Exception:
            font = None
    if font is None:
        try:
            # DejaVu often exists on systems; otherwise fallback to default.
            font = ImageFont.truetype("DejaVuSans.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()
    # measure and center text horizontally/vertically inside bbox
    w, h = draw.textsize(new_name, font=font)
    tx = x1 + max(0, (x2 - x1 - w) // 2)
    ty = y1 + max(0, (y2 - y1 - h) // 2)
    draw.text((tx, ty), new_name, font=font, fill=tuple(color))
    return pil_img

def resize_for_ocr(pil: Image.Image, max_dim: int = 1280):
    """
    Return (resized_pil, scale) where scale = resized_dim / original_dim.
    We will multiply OCR coords by (1/scale) to map back to original.
    """
    w, h = pil.size
    if max(w, h) <= max_dim:
        return pil, 1.0
    scale = max_dim / float(max(w, h))
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = pil.resize((new_w, new_h), Image.LANCZOS)
    return resized, scale

async def personalize_page(original_page: Path, output_page: Path, new_name: str, meta_for_page: Optional[dict]=None, story_dir: Optional[Path]=None):
    """
    Personalized page creation:
      - if meta_for_page present: use provided bbox/font/etc
      - else: use OCR (on downscaled image) to detect text region, erase and draw new_name
      - if everything fails, copy original page (no change)
    """
    try:
        pil_orig = Image.open(original_page).convert("RGB")
    except Exception as e:
        logger.error("Failed to open %s: %s", original_page, e)
        shutil.copyfile(original_page, output_page)
        return str(output_page)

    bbox = None
    font_path = None
    font_size = None
    color = (0,0,0)

    # Use meta if available
    if meta_for_page:
        try:
            bb = meta_for_page.get("name_bbox")
            if bb and len(bb) == 4:
                bbox = (int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3]))
            font_path = (story_dir / meta_for_page.get("font")) if meta_for_page.get("font") else None
            font_size = int(meta_for_page.get("font_size", 48))
            color = tuple(meta_for_page.get("font_color", (0,0,0)))
        except Exception:
            bbox = None

    # If no meta bbox, use OCR (but run OCR on a downscaled image to save RAM)
    if bbox is None and OCR_READER:
        try:
            pil_for_ocr, scale = resize_for_ocr(pil_orig, max_dim=1280)
            img_np = np.array(pil_for_ocr)
            ocr_results = OCR_READER.readtext(img_np)  # list of (bbox, text, confidence)
            best = choose_best_text_candidate(ocr_results, new_name)
            if best:
                _, obbox, text_found, conf = best
                # obbox coords are in resized image space; map them back to original coordinates
                x1_s, y1_s, x2_s, y2_s = bbox_to_xy(obbox)
                # scale factor: resized = original * scale => original = resized / scale
                inv_scale = 1.0 / scale if scale != 0 else 1.0
                bbox = (int(x1_s * inv_scale), int(y1_s * inv_scale), int(x2_s * inv_scale), int(y2_s * inv_scale))
                # heuristics for font size and color
                font_size = int(bbox[3] - bbox[1])
                # choose text color (simple heuristic) -> default dark
                color = (10,10,10)
        except Exception as e:
            logger.warning("OCR failed for %s: %s", original_page, e)
            bbox = None

    # If still no bbox, fallback: copy original page
    if bbox is None:
        logger.info("No name bbox detected for %s; copying original.", original_page)
        shutil.copyfile(original_page, output_page)
        return str(output_page)

    # erase and draw new name on the original-size image
    try:
        pil_out = pil_orig.copy()
        pil_out = erase_bbox(pil_out, bbox)
        pil_out = draw_name(pil_out, bbox, new_name, font_path=str(font_path) if font_path else None, font_size=font_size, color=color)
        pil_out.save(output_page, format="PNG")
        return str(output_page)
    except Exception as e:
        logger.exception("Failed to personalize page %s: %s", original_page, e)
        shutil.copyfile(original_page, output_page)
        return str(output_page)

# ---------- PDF assembler (unchanged) ----------
async def _assemble_images_to_pdf(image_paths: List[Path], pdf_path: Path) -> None:
    """Use PIL to assemble images into a multi-page PDF (run blocking work in threadpool)."""
    loop = asyncio.get_running_loop()

    def do_create():
        imgs = []
        for p in image_paths:
            im = Image.open(p).convert("RGB")
            imgs.append(im)
        if not imgs:
            raise ValueError("No images passed to PDF assembler")
        first, rest = imgs[0], imgs[1:]
        first.save(pdf_path, save_all=True, append_images=rest)

    await loop.run_in_executor(None, do_create)

# ---------- Endpoint: generate-book (two-phase) ----------
@app.post("/api/generate-book")
async def generate_book(
    file: UploadFile = File(...),
    name: str = Form(...),
    gender: Gender = Form(...),
    story: StoryChoice = Form(...),
):
    """
    Two-phase flow:
      1) Personalize all pages (replace names) -> saved to storage/stories/<story>/new_pages/
      2) Run face-swap on personalized pages -> save to storage/output/swapped_pages/
      3) Assemble swapped pages to PDF -> storage/output/pdfs/
      4) Upload swapped pages + PDF to R2 (if configured) and return cloud URLs
    """
    # validate file ext & size
    ext = _ext_from_filename(file.filename)
    if ext not in ALLOWED_EXT:
        raise HTTPException(status_code=400, detail="Unsupported file type. Use png/jpg/jpeg.")

    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(status_code=400, detail=f"File too large. Max {MAX_FILE_SIZE_BYTES} bytes.")

    # save uploaded face for re-use across pages
    uid = uuid.uuid4().hex
    input_filename = f"{uid}{ext}"
    input_path = INPUT_DIR / input_filename
    await save_bytes_to_path(contents, input_path)
    logger.info("Saved uploaded face to %s", input_path)

    # gather pages for chosen story/gender (original templates)
    pages_parent = STORIES_DIR / story.value / gender.value
    if not pages_parent.exists():
        logger.error("Story path not found: %s", pages_parent)
        raise HTTPException(status_code=404, detail=f"Story path not found: {pages_parent}")

    pages = sorted([p for p in pages_parent.iterdir() if p.suffix.lower() in ALLOWED_EXT], key=lambda p: natural_sort_key(p.name))
    if not pages:
        logger.error("No pages found for story=%s gender=%s", story.value, gender.value)
        raise HTTPException(status_code=404, detail="No story pages found for requested story/gender.")

    # PHASE 1: Personalize all pages → save to storage/stories/<story>/new_pages/
    new_pages_dir = STORIES_DIR / story.value / "new_pages"
    os.makedirs(new_pages_dir, exist_ok=True)

    # If a meta.json present for the story, use it
    story_meta = load_story_meta(STORIES_DIR / story.value)

    personalized_paths: List[Path] = []
    for idx, page_path in enumerate(pages, start=1):
        out_name = f"{idx:03}_personalized.png"
        out_path = new_pages_dir / out_name
        # find metadata for this page if any
        meta_for_page = None
        if story_meta:
            for m in story_meta.get("pages", []):
                if m.get("filename") == page_path.name:
                    meta_for_page = m
                    break
        saved = await personalize_page(page_path, out_path, name, meta_for_page, story_dir=STORIES_DIR / story.value)
        personalized_paths.append(Path(saved))
        logger.info("Personalized page %d -> %s", idx, saved)

    logger.info("Phase 1 complete: %d personalized pages created at %s", len(personalized_paths), new_pages_dir)

    # PHASE 2: Run face-swap on personalized pages → save to OUTPUT_DIR
    swapped_paths: List[Path] = []
    for idx, ppage in enumerate(personalized_paths, start=1):
        out_name = f"{uid}_{idx:03}.png"
        out_path = OUTPUT_DIR / out_name
        logger.info("Face-swapping page %d: %s -> %s", idx, ppage, out_path)
        saved = await create_piapi_task_and_wait(ppage, input_path, out_path)
        if saved:
            swapped_paths.append(Path(saved))
        else:
            logger.info("Fallback: copying personalized page for idx %d", idx)
            shutil.copyfile(ppage, out_path)
            swapped_paths.append(out_path)

    # assemble pdf (ensure swapped_paths are sorted by filename index)
    swapped_paths = sorted(swapped_paths, key=lambda p: natural_sort_key(p.name))
    pdf_name = f"{uid}_{story.value}_{gender.value}.pdf"
    pdf_path = PDF_DIR / pdf_name
    try:
        await _assemble_images_to_pdf(swapped_paths, pdf_path)
    except Exception as e:
        logger.exception("Failed to create PDF: %s", e)
        raise HTTPException(status_code=500, detail="Failed to assemble PDF.")

    # UPLOAD to R2 if configured, else use local /assets URLs
    r2_client = get_r2_client()
    swapped_urls: List[str] = []
    pdf_url: str

    if r2_client:
        logger.info("R2 configured — uploading swapped pages and PDF to R2.")
        # upload swapped pages
        for p in swapped_paths:
            remote_key = f"output/swapped_pages/{p.name}"
            uploaded = upload_to_r2(p, remote_key)
            if uploaded:
                swapped_urls.append(uploaded)
                try:
                    # remove local swapped file to save disk
                    os.remove(p)
                except Exception:
                    pass
            else:
                # fallback to local assets url
                swapped_urls.append(f"{ASSETS_URL_PREFIX}/output/swapped_pages/{p.name}")

        # upload pdf
        pdf_remote_key = f"output/pdfs/{pdf_path.name}"
        uploaded_pdf = upload_to_r2(pdf_path, pdf_remote_key)
        if uploaded_pdf:
            pdf_url = uploaded_pdf
            try:
                os.remove(pdf_path)
            except Exception:
                pass
        else:
            pdf_url = f"{ASSETS_URL_PREFIX}/output/pdfs/{pdf_name}"
    else:
        logger.info("R2 not configured — returning local asset URLs.")
        swapped_urls = [f"{ASSETS_URL_PREFIX}/output/swapped_pages/{p.name}" for p in swapped_paths]
        pdf_url = f"{ASSETS_URL_PREFIX}/output/pdfs/{pdf_name}"

    return JSONResponse({
        "status": "ok",
        "kid_name": name,
        "story": story.value,
        "gender": gender.value,
        "personalized_pages_dir": str(new_pages_dir),
        "swapped_pages": swapped_urls,
        "output_pdf": pdf_url
    })

# ---------- Health ----------
@app.get("/")
async def root():
    return {
        "status": "alive",
        "hint": "POST file+name+gender+story to /api/generate-book. Use /docs for interactive testing."
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
