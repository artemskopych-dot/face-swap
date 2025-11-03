import base64
import logging
import cv2
import numpy as np
import requests
from typing import Tuple
from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import JSONResponse
from openai import OpenAI

LOGGER = logging.getLogger("human_face_swap")

app = FastAPI(
    title="OpenAI Human Face Swap Endpoint",
    version="2.0.0",
    description="Takes target and reference image URLs and performs a face swap via OpenAI image editing API.",
)

def _download_image(url: str) -> np.ndarray:
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download image from {url}: {e}")
    buffer = np.frombuffer(resp.content, dtype=np.uint8)
    img = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail=f"Cannot decode image from {url}")
    return img

def _encode_png(image: np.ndarray) -> bytes:
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise RuntimeError("Failed to encode PNG image")
    return encoded.tobytes()

def _b64encode(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")

def _create_face_mask(image: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if cascade.empty():
        raise RuntimeError("OpenCV cascade classifier could not be loaded")

    faces = cascade.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=6, minSize=(80, 80))
    if len(faces) == 0:
        raise HTTPException(status_code=422, detail="No recognizable face located in the target image.")

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    pad = int(max(w, h) * 0.25)
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(image.shape[1], x + w + pad)
    y2 = min(image.shape[0], y + h + pad)

    mask = np.ones((image.shape[0], image.shape[1], 4), dtype=np.uint8) * 255
    mask[y1:y2, x1:x2, 3] = 0
    bbox = (int(x1), int(y1), int(x2), int(y2))
    return mask, bbox

@app.post("/face-swap")
async def swap_face(
    target_url: str = Form(...),
    reference_url: str = Form(...),
    style_prompt: str = Form(""),
    openai_api_key: str = Form(...),
) -> JSONResponse:

    target_np = _download_image(target_url)
    reference_np = _download_image(reference_url)

    try:
        mask_np, bbox = _create_face_mask(target_np)
    except HTTPException:
        raise
    except Exception as exc:
        LOGGER.exception("Mask creation failed: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to compute face mask for target image.")

    target_png = _encode_png(target_np)
    mask_png = _encode_png(mask_np)
    reference_png = _encode_png(reference_np)

    client = OpenAI(api_key=openai_api_key)

    data_url = f"data:image/png;base64,{_b64encode(reference_png)}"
    try:
        response = client.responses.create(
            model="gpt-4.1-mini",
            temperature=0.2,
            input=[
                {
                    "role": "system",
                    "content": (
                        "You describe human faces precisely and concisely, focusing on traits that make them recognizable."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "Describe the person's facial characteristics, "
                                "including perceived age range, skin tone, hair style and color, "
                                "eye color, freckles or birthmarks, and any standout traits in two brief sentences. "
                                "Do not guess their name."
                            ),
                        },
                        {"type": "input_image", "image_url": data_url, "detail": "high"},
                    ],
                },
            ],
        )
        reference_description = (response.output_text or "").strip()
    except Exception as exc:
        LOGGER.exception("OpenAI face description failed: %s", exc)
        raise HTTPException(status_code=502, detail="OpenAI face description request failed.")

    prompt_parts = [
        "Replace only the face inside the provided mask with the reference person from the photo.",
        "Preserve the original artwork style, lighting, and palette.",
        "Blend the new face naturally while keeping key distinctive facial features.",
        reference_description,
    ]
    if style_prompt.strip():
        prompt_parts.append(style_prompt.strip())
    composed_prompt = " ".join(prompt_parts)

    try:
        response_edit = client.images.edit(
            model="gpt-image-1",
            image=("target.png", target_png, "image/png"),
            mask=("mask.png", mask_png, "image/png"),
            prompt=composed_prompt,
            size="auto",
            quality="high",
        )
        swapped_b64 = response_edit.data[0].b64_json
    except Exception as exc:
        LOGGER.exception("OpenAI image edit failed: %s", exc)
        raise HTTPException(status_code=502, detail="OpenAI image edit request failed.")

    return JSONResponse({
        "swapped_image_png_b64": swapped_b64,
        "prompt": composed_prompt,
        "reference_description": reference_description,
        "mask_png_b64": _b64encode(mask_png),
        "mask_bbox": {"x1": bbox[0], "y1": bbox[1], "x2": bbox[2], "y2": bbox[3]},
    })
