# app.py (updated for Kaggle-style RGB output)
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import tensorflow as tf
import nibabel as nib
import tempfile, os, io, traceback
from PIL import Image
import cv2

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# CONFIG
MODEL_PATH = "tumor_segmentation_model.h5"
TARGET_SIZE = (128, 128)
SLICE_STEP = 2
LIVER_WINDOW = (150, 30)
CUSTOM_WINDOW = (200, 60)

# --- CHANGED: Kaggle-style RGB color map ---
CLASS_COLOR_MAP = {
    0: (0, 0, 0),        # background - black
    1: (255, 255, 255),  # liver - white
    2: (255, 0, 0)       # tumor - red
}

# load model
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("Loaded model:", MODEL_PATH)
print("Model input shape:", model.input_shape, "output shape:", model.output_shape)

def apply_window(slice2d: np.ndarray, width: int, level: int) -> np.ndarray:
    px = slice2d.astype(np.float32).copy()
    low = level - width / 2.0
    high = level + width / 2.0
    px[px < low] = low
    px[px > high] = high
    px = (px - low) / (high - low + 1e-8)
    px = (px * 255.0).astype(np.uint8)
    return px

def freqhist_bins(arr_flat: np.ndarray, n_bins:int=100) -> np.ndarray:
    s = np.sort(arr_flat)
    t = np.concatenate([[0.001], np.arange(n_bins)/n_bins + (1/(2*n_bins)), [0.999]])
    idx = (t * len(s)).astype(int)
    idx = np.clip(idx, 0, len(s)-1)
    brks = np.unique(s[idx])
    return brks

def hist_scaled(slice2d: np.ndarray, brks=None, n_bins:int=100) -> np.ndarray:
    flat = slice2d.flatten()
    if brks is None:
        brks = freqhist_bins(flat, n_bins=n_bins)
    ys = np.linspace(0.0, 1.0, len(brks))
    scaled = np.interp(flat, brks, ys)
    scaled = scaled.reshape(slice2d.shape).astype(np.float32)
    return np.clip(scaled, 0.0, 1.0)

def to_3channel(slice2d: np.ndarray) -> np.ndarray:
    ch0 = apply_window(slice2d, width=LIVER_WINDOW[0], level=LIVER_WINDOW[1])
    ch1 = apply_window(slice2d, width=CUSTOM_WINDOW[0], level=CUSTOM_WINDOW[1])
    ch2 = (hist_scaled(slice2d, n_bins=100) * 255.0).astype(np.uint8)
    stacked = np.stack([ch0, ch1, ch2], axis=-1)
    return stacked

def preprocess_volume(nifti_path: str):
    img = nib.load(nifti_path)
    arr = img.get_fdata()
    arr = np.rot90(np.array(arr))
    H, W, D = arr.shape
    processed = []
    slice_idxs = []
    for i in range(0, D, SLICE_STEP):
        slice2d = arr[..., i].astype(np.float32)
        ch3 = to_3channel(slice2d)
        pil = Image.fromarray(ch3, mode="RGB")
        pil = pil.resize(TARGET_SIZE, resample=Image.BILINEAR)
        arr_in = (np.asarray(pil).astype(np.float32) / 255.0).astype(np.float32)
        processed.append(np.expand_dims(arr_in, axis=0))
        slice_idxs.append(i)
    return processed, slice_idxs, arr.shape

# --- CHANGED: Post-processing now returns RGB mask ---
def postprocess_prediction_slices(pred_slices: dict, slice_idxs: list, original_shape: tuple):
    H_orig, W_orig, D = original_shape
    mask_vol = np.zeros((H_orig, W_orig, D, 3), dtype=np.uint8)  # RGB volume

    for idx in slice_idxs:
        pred = pred_slices[idx]
        if pred.ndim == 3 and pred.shape[-1] > 1:
            class_map = np.argmax(pred, axis=-1).astype(np.uint8)
        elif pred.ndim == 2:
            class_map = (pred > 0.5).astype(np.uint8)
        else:
            class_map = np.argmax(pred, axis=-1).astype(np.uint8)

        rgb_mask = np.zeros((class_map.shape[0], class_map.shape[1], 3), dtype=np.uint8)
        for k, color in CLASS_COLOR_MAP.items():
            rgb_mask[class_map == k] = color

        rgb_up = cv2.resize(rgb_mask, (W_orig, H_orig), interpolation=cv2.INTER_NEAREST)
        mask_vol[..., idx, :] = rgb_up

    return mask_vol

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".nii") as tmp:
            tmp.write(contents); tmp.flush(); tmp_path = tmp.name
        try:
            processed_list, slice_idxs, orig_shape = preprocess_volume(tmp_path)
            pred_slices = {}
            for arr, idx in zip(processed_list, slice_idxs):
                p = model.predict(arr, verbose=0)
                if p.ndim == 4:
                    p = p[0]
                pred_slices[idx] = p

            mask_vol = postprocess_prediction_slices(pred_slices, slice_idxs, orig_shape)

            # --- CHANGED: return RGB mask for middle slice ---
            mid = orig_shape[2] // 2
            png = Image.fromarray(mask_vol[..., mid, :].astype(np.uint8), mode="RGB")
            buf = io.BytesIO(); png.save(buf, format="PNG"); buf.seek(0)
            return Response(content=buf.getvalue(), media_type="image/png")

        finally:
            try: os.remove(tmp_path)
            except: pass
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
