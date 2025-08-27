# app.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import numpy as np
import tensorflow as tf
import nibabel as nib
import tempfile, os, io, traceback
from PIL import Image, ImageOps
import cv2

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === CONFIG - change if needed ===
MODEL_PATH = "tumor_segmentation_model.h5"   # ensure correct path
TARGET_SIZE = (128, 128)                     # model trained size
SLICE_STEP = 2                               # training used step=2 -> replicate that
LIVER_WINDOW = (150, 30)                     # (width, level) as in notebook example
CUSTOM_WINDOW = (200, 60)                    # adjust if notebook used different custom
CLASS_COLOR_MAP = {0: 0, 1: 128, 2: 255}     # background=0, liver=128, tumor=255
# ==================================

# Load model (compile=False to avoid custom objects requirement)
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("Loaded model:", MODEL_PATH)
print("Model input shape:", model.input_shape, "output shape:", model.output_shape)


# ------------------ helper functions (match training) ------------------ #
def apply_window(slice2d: np.ndarray, width: int, level: int) -> np.ndarray:
    """Apply windowing (HU style): clamp to [level-width/2, level+width/2], scale to 0-255 uint8"""
    px = slice2d.astype(np.float32).copy()
    low = level - width / 2.0
    high = level + width / 2.0
    px[px < low] = low
    px[px > high] = high
    # scale to 0-255
    px = (px - low) / (high - low + 1e-8)
    px = (px * 255.0).astype(np.uint8)
    return px

def freqhist_bins(arr_flat: np.ndarray, n_bins:int=100) -> np.ndarray:
    """Approx of the notebook's freqhist_bins. Returns breakpoints"""
    # sort flatten
    s = np.sort(arr_flat)
    # create percentile indices similar to original
    t = np.concatenate([[0.001], np.arange(n_bins)/n_bins + (1/(2*n_bins)), [0.999]])
    idx = (t * (len(s))).astype(int)
    idx = np.clip(idx, 0, len(s)-1)
    brks = np.unique(s[idx])
    return brks

def hist_scaled(slice2d: np.ndarray, brks=None, n_bins:int=100) -> np.ndarray:
    """Map intensities to 0..1 using freqhist binning (like notebook). Return float32 0..1."""
    flat = slice2d.flatten()
    if brks is None:
        brks = freqhist_bins(flat, n_bins=n_bins)
    # ys = linspace(0,1,len(brks))
    ys = np.linspace(0.0, 1.0, len(brks))
    # Interpolate
    scaled = np.interp(flat, brks, ys)
    scaled = scaled.reshape(slice2d.shape).astype(np.float32)
    return np.clip(scaled, 0.0, 1.0)

def to_3channel(slice2d: np.ndarray, wins=[LIVER_WINDOW, CUSTOM_WINDOW]) -> np.ndarray:
    """
    Build 3-channel image as in notebook:
      channel0 = liver window (0-255)
      channel1 = custom window (0-255)
      channel2 = hist_scaled (0-1) -> then scaled to 0-255
    Returns uint8 (H,W,3)
    """
    chs = []
    for w in wins:
        ch = apply_window(slice2d, width=w[0], level=w[1])
        chs.append(ch)
    # hist scaled channel
    hscaled = (hist_scaled(slice2d, n_bins=100) * 255.0).astype(np.uint8)
    chs.append(hscaled)
    # stack -> H,W,3
    stacked = np.stack(chs, axis=-1)
    return stacked  # uint8

def preprocess_volume(nifti_path: str):
    """
    Read NIfTI, rotate (np.rot90) like notebook, then for each slice in range(0, depth, SLICE_STEP)
    produce normalized (0..1) numpy arrays resized to TARGET_SIZE and also store original slices for overlay.
    Returns:
      - processed_list: list of arrays shape (1,H,W,3) float32 in 0..1 for model
      - slice_indices: list of original indices corresponding to processed slices
      - original_slices: dict idx -> original 2D array (float) (for potential overlay or size info)
      - original_shape: (H_orig, W_orig, D)
    """
    img = nib.load(nifti_path)
    arr = img.get_fdata()  # float
    # rotate same as notebook
    arr = np.rot90(np.array(arr))
    H, W, D = arr.shape
    processed = []
    slice_idxs = []
    orig_slices = {}
    for i in range(0, D, SLICE_STEP):
        slice2d = arr[..., i].astype(np.float32)
        orig_slices[i] = slice2d  # keep original (float)
        # create 3-channel uint8 as training saved JPGs
        ch3 = to_3channel(slice2d)  # uint8 HxWx3
        # convert to PIL, resize with BILINEAR
        pil = Image.fromarray(ch3, mode="RGB")
        pil = pil.resize(TARGET_SIZE, resample=Image.BILINEAR)
        arr_in = (np.asarray(pil).astype(np.float32) / 255.0).astype(np.float32)  # H,W,3 in 0..1
        # model expects batch dim
        processed.append(np.expand_dims(arr_in, axis=0))
        slice_idxs.append(i)
    return processed, slice_idxs, orig_slices, arr.shape

def postprocess_prediction_slices(pred_slices: dict, slice_idxs: list, original_shape: tuple):
    """
    pred_slices: dict idx-> prediction array (H_model,W_model,C) or (H_model,W_model)
    Fill a 3D volume of same depth as original with class ids (0..2) at the processed indices.
    Return: 3D uint8 mask volume (H_orig x W_orig x D) with values mapped to CLASS_COLOR_MAP (0/128/255).
    """
    H_orig, W_orig, D = original_shape
    # start with zeros
    mask_vol = np.zeros((H_orig, W_orig, D), dtype=np.uint8)
    for idx in slice_idxs:
        pred = pred_slices[idx]
        # pred could be (H,W,C) with probabilities
        if pred.ndim == 3 and pred.shape[-1] > 1:
            class_map = np.argmax(pred, axis=-1).astype(np.uint8)  # H_model x W_model
        elif pred.ndim == 2:
            # binary single channel - threshold
            p = pred
            th = 0.5
            class_map = (p > th).astype(np.uint8)
        else:
            class_map = np.argmax(pred, axis=-1).astype(np.uint8)
        # map classes to grayscale intensities
        vis = np.zeros_like(class_map, dtype=np.uint8)
        for k, v in CLASS_COLOR_MAP.items():
            vis[class_map == k] = v
        # resize vis back to original slice size using INTER_NEAREST (keep labels)
        vis_up = cv2.resize(vis, (W_orig, H_orig), interpolation=cv2.INTER_NEAREST)
        mask_vol[..., idx] = vis_up
    return mask_vol

def make_mask_png_for_middle(mask_vol: np.ndarray, orig_slices: dict):
    """
    Create a PNG (bytes) of the middle slice mask (mapped grayscale).
    Return PNG bytes.
    """
    H, W, D = mask_vol.shape
    mid = D // 2
    # if mid slice is empty (zeros), try nearest non-empty slice
    if mask_vol[..., mid].max() == 0:
        # search nearby
        offsets = list(range(1, D//2 + 1))
        found = False
        for off in offsets:
            if mid - off >= 0 and mask_vol[..., mid-off].max() > 0:
                mid = mid - off
                found = True
                break
            if mid + off < D and mask_vol[..., mid+off].max() > 0:
                mid = mid + off
                found = True
                break
    mask_img = mask_vol[..., mid].astype(np.uint8)
    pil = Image.fromarray(mask_img, mode="L")
    byte_arr = io.BytesIO()
    pil.save(byte_arr, format="PNG")
    byte_arr.seek(0)
    return byte_arr.getvalue(), mid

# ------------------ API endpoint ------------------ #
ALLOWED_MIME = ["application/octet-stream", "application/zip", "application/x-nifti", "application/vnd.nifti", "application/dicom", "application/x-gzip"]

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accepts a .nii (or similar) file upload.
    - Internally preprocesses every SLICE_STEP slice (0..D step SLICE_STEP) exactly like training
      (rotation, liver+custom+hist_scaled -> 3-channel -> PIL resize -> /255)
    - Runs model.predict on all processed slices
    - Builds a 3D mask volume aligning to original depth positions
    - Returns a single PNG (middle slice mask) to the client for quick visualization
    """
    try:
        if file.content_type not in ALLOWED_MIME:
            # still allow but warn
            print("Warning: content_type:", file.content_type)
        contents = await file.read()
        # save to temp
        with tempfile.NamedTemporaryFile(delete=False, suffix=".nii") as tmp:
            tmp.write(contents)
            tmp.flush()
            tmp_path = tmp.name

        try:
            # Preprocess -> list of (1,H,W,3) arrays and their slice indices
            processed_list, slice_idxs, orig_slices, orig_shape = preprocess_volume(tmp_path)
            print(f"Preprocessed {len(processed_list)} slices (step={SLICE_STEP}). Example indices:", slice_idxs[:6])

            # Predict each slice (we do sequential predict to avoid large memory)
            pred_slices = {}
            for arr, idx in zip(processed_list, slice_idxs):
                p = model.predict(arr, verbose=0)  # p shape: (1,H_model,W_model,C) or (1,H_model,W_model)
                if p.ndim == 4:
                    p = p[0]
                elif p.ndim == 3 and p.shape[0] == 1:
                    p = p[0]
                pred_slices[idx] = p  # store
            print("Predictions done for all processed slices.")

            # Build 3D mask volume aligned with original
            mask_vol = postprocess_prediction_slices(pred_slices, slice_idxs, orig_shape)
            print("Mask volume built. shape:", mask_vol.shape)

            # Produce PNG of middle slice mask for quick feedback
            png_bytes, used_mid = make_mask_png_for_middle(mask_vol, orig_slices)
            print(f"Returning middle slice mask at index {used_mid}")

            return Response(content=png_bytes, media_type="image/png")

        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
