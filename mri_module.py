"""
Breast MRI Analysis — standalone module.

Fully separate from the existing ultrasound, PET, and mammography
pipelines: no shared model class, preprocessing, or prediction function.
The MRI model is never used on any other modality's images and vice versa.

Architecture: DenseNet201 (ImageNet-pretrained) + a small classification
head, fine-tuned via 2-stage transfer learning on the breast MRI dataset
(see train_mri_kaggle.py, run on a Kaggle Notebook — the dataset is never
downloaded to this project).

To connect a freshly trained model:
  1. Run train_mri_kaggle.py in a Kaggle Notebook.
  2. Download the resulting `mri_model.pt` from Kaggle's Output panel.
  3. Place it next to main.py (same folder as model_cnn.pt).
  That's it — this module picks it up automatically on next app start,
  including the real class names it was trained with (no invented labels).
"""

import os
import tempfile
from datetime import datetime, timezone
from typing import Optional

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
try:
    from organ_detector import predict_organ, is_breast_label
except Exception:
    predict_organ = None
    is_breast_label = None

try:
    from torchvision.models import densenet201, DenseNet201_Weights
except Exception:  # pragma: no cover
    densenet201 = None
    DenseNet201_Weights = None

# ---------------------------------------------------------------------------
# 1) MODEL DEFINITION (must match train_mri_kaggle.py exactly)
# ---------------------------------------------------------------------------

MRI_MODEL_PATH = "mri_model.pt"
DEFAULT_CLASS_NAMES = ["benign", "malignant"]  # fallback only; real run uses
                                                # the names saved in mri_model.pt


class MRIClassifier(nn.Module):
    """Same architecture as in train_mri_kaggle.py. Defined here (not
        imported from main.py or the mammography module) to keep this module
    fully independent."""

    def __init__(self, num_classes: int):
        super().__init__()
        backbone = densenet201(weights=None) if densenet201 is not None else None
        self.features = backbone.features
        num_backbone_features = backbone.classifier.in_features
        self.head = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(num_backbone_features, 1024),
            nn.SiLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.SiLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))

    def forward_features(self, x):
        return self.features(x)

    def forward_from_features(self, feat):
        return self.head(feat)


# ---------------------------------------------------------------------------
# 2) MODEL LOADING
# ---------------------------------------------------------------------------

_MRI_MODEL = None
_MRI_CLASS_NAMES = None
_MRI_LOAD_ATTEMPTED = False


def load_mri_model(path: str = MRI_MODEL_PATH):
    if densenet201 is None or not os.path.exists(path):
        return None, None
    try:
        checkpoint = torch.load(path, map_location="cpu")
        class_names = checkpoint.get("class_names", DEFAULT_CLASS_NAMES)
        model = MRIClassifier(num_classes=len(class_names))
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        return model, class_names
    except Exception:
        return None, None


def get_mri_model():
    global _MRI_MODEL, _MRI_CLASS_NAMES, _MRI_LOAD_ATTEMPTED
    if not _MRI_LOAD_ATTEMPTED:
        _MRI_LOAD_ATTEMPTED = True
        _MRI_MODEL, _MRI_CLASS_NAMES = load_mri_model()
    return _MRI_MODEL, _MRI_CLASS_NAMES


def mri_model_status_text() -> str:
    if os.path.exists(MRI_MODEL_PATH):
        return (
            "Model file detected (mri_model.pt). It will be loaded only when "
            "you run an MRI analysis."
        )
    return (
           "No trained breast MRI model connected yet — showing a placeholder "
           "analysis only. Run 'python train_mri_local.py' "
           "from the project folder, then restart the app to enable real "
           "predictions."
    )


# ---------------------------------------------------------------------------
# 3) PREPROCESSING (must match eval_transform in the training script)
# ---------------------------------------------------------------------------

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def preprocess_mri(pil_img: Image.Image) -> torch.Tensor:
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return transform(pil_img.convert("RGB")).unsqueeze(0)


# ---------------------------------------------------------------------------
# 4) GRAD-CAM (last conv layer of the DenseNet201 backbone)
# ---------------------------------------------------------------------------

def mri_grad_cam(model: MRIClassifier, input_tensor: torch.Tensor):
    model.zero_grad()
    feat = model.forward_features(input_tensor)
    feat.retain_grad()
    logits = model.forward_from_features(feat)
    target_class = int(logits.argmax(dim=1).item())
    logits[0, target_class].backward()

    gradient = feat.grad
    activation = feat.detach()
    if activation is None or gradient is None:
        return None, "Gradient computation unavailable for this model."

    weights = gradient.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * activation).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False)
    cam = cam.squeeze().detach().cpu().numpy()
    cam_min, cam_max = float(cam.min()), float(cam.max())
    if cam_max - cam_min < 1e-6:
        return None, "Model attention was uniform across the image for this input."
    cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
    return cam, None


def gradcam_overlay(img: Image.Image, cam: np.ndarray, alpha: float = 0.5) -> Image.Image:
    import matplotlib
    try:
        cmap = matplotlib.colormaps["jet"]
    except Exception:
        import matplotlib.cm as cm
        cmap = cm.get_cmap("jet")
    colored = (cmap(cam)[:, :, :3] * 255).astype(np.uint8)
    heat = Image.fromarray(colored).convert("RGBA").resize(img.size, resample=Image.BILINEAR)
    return Image.blend(img.convert("RGBA"), heat, alpha=alpha)


# ---------------------------------------------------------------------------
# 5) PREDICTION
# ---------------------------------------------------------------------------

def predict_mri(image: np.ndarray):
    if image is None:
        return {
            "label": "No image uploaded",
            "confidence": None,
            "detail": "Please upload an MRI image first.",
            "report_path": None,
            "is_placeholder": True,
            "explanation_img": None,
        }

    pil = Image.fromarray(np.uint8(image)).convert("RGB")

    # Enforce strict gating: organ detector if present + a strict MRI
    # heuristic. Reject unconditionally if either indicates non-breast.
    # Require organ_detector to be present to avoid accidental non-breast analysis.
    if predict_organ is None:
        fd, report_path = tempfile.mkstemp(suffix=".txt", prefix="mri_report_")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write("Breast MRI Analysis Report\n\n")
            f.write("Assessment: REJECTED — organ detector not available on this installation.\n")
            f.write("Place a trained 'organ_detector.pt' next to main.py to enable safe gating.\n")
        return {
            "label": "REJECTED - organ_detector missing",
            "confidence": None,
            "detail": "organ_detector not installed; analysis disabled to avoid non-breast predictions.",
            "report_path": report_path,
            "is_placeholder": True,
            "explanation_img": None,
        }

    organ_label, organ_conf = ("unknown", 0.0)
    try:
        organ_label, organ_conf = predict_organ(pil)
    except Exception:
        organ_label, organ_conf = "unknown", 0.0

    def is_strict_breast_mri(img: Image.Image) -> (bool, str):
        gray = np.asarray(img.convert("L"), dtype=np.float32)
        h, w = gray.shape

        # require a non-trivial, non-total tissue fraction (rejects near-blank
        # and near-fully-saturated images alike)
        tissue_frac = float((gray > 10).sum()) / (h * w)
        if tissue_frac < 0.02:
            return False, "Too little tissue-like area for a breast MRI."
        if tissue_frac > 0.98:
            return False, "Image is almost entirely uniform — unlikely an MRI."

        std = float(gray.std())
        if std < 12.0 or std > 120.0:
            return False, "Image contrast/variance outside expected MRI range."

        # MRI slices are effectively grayscale even when saved as RGB —
        # reject images with real color content (photos, skin, other scans
        # rendered in color).
        arr_rgb = np.asarray(img.convert("RGB"), dtype=np.float32)
        r, g, b = arr_rgb[..., 0], arr_rgb[..., 1], arr_rgb[..., 2]
        channel_spread = float(np.mean(np.abs(r - g)) + np.mean(np.abs(g - b)) + np.mean(np.abs(r - b)))
        if channel_spread > 14.0:
            return False, "Image has too much color to be an MRI scan."

        # MRI slices typically show a roughly centered anatomical cross-section
        # against a dark background, not a subject filling one edge of frame.
        bright_thresh = np.percentile(gray, 65)
        ys, xs = np.where(gray > bright_thresh)
        if ys.size == 0:
            return False, "No tissue-like region detected."
        center_y = float(ys.mean()) / h
        center_x = float(xs.mean()) / w
        if center_y < 0.05 or center_y > 0.95 or center_x < 0.05 or center_x > 0.95:
            return False, "Tissue location atypical for a breast MRI slice."

        return True, ""

    heuristic_ok, heuristic_reason = is_strict_breast_mri(pil)

    if organ_conf >= 0.6 and organ_label != "mri":
        fd, report_path = tempfile.mkstemp(suffix=".txt", prefix="mri_report_")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write("Breast MRI Analysis Report\n\n")
            f.write("Rejected: uploaded image does not appear to be a breast MRI or breast image.\n")
            f.write(f"Detected as: {organ_label} (confidence {organ_conf:.2f})\n")
        return {
            "label": "REJECTED - Not a breast MRI",
            "confidence": None,
            "detail": "Uploaded image does not appear to be a breast MRI or breast image.",
            "report_path": report_path,
            "is_placeholder": True,
            "explanation_img": None,
        }

    if not heuristic_ok:
        fd, report_path = tempfile.mkstemp(suffix=".txt", prefix="mri_report_")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write("Breast MRI Analysis Report\n\n")
            f.write("Rejected: uploaded image does not appear to be a breast MRI or breast image.\n")
            f.write(f"Reason: {heuristic_reason}\n")
        return {
            "label": "REJECTED - Not a breast MRI",
            "confidence": None,
            "detail": "Uploaded image does not appear to be a breast MRI or breast image.",
            "report_path": report_path,
            "is_placeholder": True,
            "explanation_img": None,
        }
    model, class_names = get_mri_model()

    explanation_img = None
    if model is not None:
        input_tensor = preprocess_mri(pil)
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1).squeeze(0)
        pred_idx = int(torch.argmax(probs).item())
        confidence = float(probs[pred_idx].item())
        label = class_names[pred_idx]
        prob_breakdown = ", ".join(
            f"{class_names[i]}: {probs[i]*100:.1f}%" for i in range(len(class_names))
        )
        detail = (
            f"Model probabilities — {prob_breakdown}. Based on a DenseNet201 "
            f"model fine-tuned on the breast MRI dataset; evaluate this "
            f"against the reported test-set metrics before trusting it."
        )
        is_placeholder = False

        cam, cam_reason = mri_grad_cam(model, input_tensor)
        if cam is not None:
            explanation_img = gradcam_overlay(pil, cam)
        else:
            print(f"[mri grad_cam] unavailable: {cam_reason}")
    else:
        label = "MRI model not yet connected"
        confidence = None
        detail = (
            "This module is ready to display and analyze breast MRI images, "
            "but no trained MRI model has been connected yet. No prediction "
            "is being made on this image. Run 'python train_mri_local.py' "
            "from the project folder, then restart the app to enable real "
            "predictions."
        )
        is_placeholder = True

    report_lines = [
        "Breast MRI Analysis Report",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        f"Result: {label}",
        f"Confidence: {confidence if confidence is not None else 'N/A'}",
        "",
        detail,
        "",
        "AI-generated result for educational/research purposes only. "
        "This is not a medical diagnosis.",
    ]

    fd, report_path = tempfile.mkstemp(suffix=".txt", prefix="mri_report_")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    return {
        "label": label,
        "confidence": confidence,
        "detail": detail,
        "report_path": report_path,
        "is_placeholder": is_placeholder,
        "explanation_img": explanation_img,
    }


# ---------------------------------------------------------------------------
# 6) RESULT DISPLAY
# ---------------------------------------------------------------------------

def build_mri_result_html(result: dict) -> str:
    label = result["label"]
    confidence = result["confidence"]
    detail = result["detail"]

    confidence_html = ""
    if confidence is not None:
        confidence_html = f"<div class='verdict-sub'>Confidence: {confidence * 100:.1f}%</div>"

    label_lower = str(label).lower()
    if "malignant" in label_lower:
        badge_color = "#dc2626"
    elif "benign" in label_lower:
        badge_color = "#16a34a"
    else:
        badge_color = "#334155"

    return f"""
    <div class="verdict-card" style="background:linear-gradient(135deg, {badge_color}, rgba(15,23,42,0.92)); border:1px solid rgba(255,255,255,0.06); box-shadow: 0 18px 36px rgba(2,6,23,0.38), inset 0 1px 0 rgba(255,255,255,0.06);">
      <div class="assist-badge" style="background:rgba(255,255,255,0.08); border-color: rgba(255,255,255,0.12); color:#f8fafc; margin-bottom:12px;">MRI result</div>
      <div class="verdict-title">{label}</div>
      {confidence_html}
      <div class="verdict-sub" style="margin-top:8px;">{detail}</div>
      <div class="verdict-sub" style="margin-top:10px;font-size:0.8rem;opacity:0.85;">
        AI-generated result for educational/research purposes only.
        This is not a medical diagnosis.
      </div>
    </div>
    """


def analyze_mri(image):
    """Top-level function wired directly into the Gradio button click."""
    if image is None:
        html = (
            "<div class='verdict-card' style='background:#334155;'>"
            "<div class='verdict-title'>No image uploaded</div>"
            "<div class='verdict-sub'>Please upload an MRI image first.</div></div>"
        )
        return html, None, None
    result = predict_mri(image)
    html = build_mri_result_html(result)
    return html, result["explanation_img"], result["report_path"]
