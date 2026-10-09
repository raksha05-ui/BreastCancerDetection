"""
Mammography Analysis — standalone module.

Fully separate from the existing breast-ultrasound pipeline in main.py:
no shared model class, preprocessing, or prediction function. Per project
requirement #6, the ultrasound model is never used on mammograms and this
module's model is never used on ultrasound images.

Architecture: DenseNet169 (ImageNet-pretrained) + a small classification
head, fine-tuned via transfer learning on CBIS-DDSM full mammogram images
(see train_mammography_kaggle.py, run on a Kaggle Notebook — that dataset
is too large to train on locally and is never downloaded to this project).

To connect a freshly trained model:
  1. Run train_mammography_kaggle.py in a Kaggle Notebook.
  2. Download the resulting `mammography_model.pt` from Kaggle's Output panel.
  3. Place it next to main.py (same folder as model_cnn.pt).
  That's it — MAMMOGRAPHY_MODEL_PATH below already points at that filename,
  and this module will pick it up automatically on next app start.
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
    from torchvision.models import densenet169, DenseNet169_Weights
except Exception:  # pragma: no cover
    densenet169 = None
    DenseNet169_Weights = None

# ---------------------------------------------------------------------------
# 1) MODEL DEFINITION (must match train_mammography_kaggle.py exactly)
# ---------------------------------------------------------------------------

MAMMOGRAPHY_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "mammography_model.pt"
)
CLASS_NAMES = {0: "BENIGN", 1: "MALIGNANT"}  # real CBIS-DDSM dataset labels
# Mammography is an allow-list input: uncertain modality predictions are
# rejected before the mammography classifier is allowed to run.
ORGAN_DETECTOR_MIN_CONFIDENCE = 0.75


class MammographyClassifier(nn.Module):
    """Same architecture as in train_mammography_kaggle.py. Defined here
    (not imported from main.py) to keep this module fully independent of
    the ultrasound Classifier."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        # weights=None here: we're about to load our own fine-tuned state
        # dict, so there's no need to re-download ImageNet weights at
        # inference time.
        backbone = densenet169(weights=None) if densenet169 is not None else None
        self.features = backbone.features
        num_backbone_features = backbone.classifier.in_features
        self.classifier = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(num_backbone_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def forward_features(self, x):
        """Raw conv feature map, used for Grad-CAM (last conv block output,
        before the final ReLU/pool/classifier head)."""
        return self.features(x)

    def forward_from_features(self, feat):
        return self.classifier(feat)


# ---------------------------------------------------------------------------
# 2) MODEL LOADING
# ---------------------------------------------------------------------------

_MAMMO_MODEL = None
_MAMMO_LOAD_ATTEMPTED = False


def load_mammography_model(path: str = MAMMOGRAPHY_MODEL_PATH) -> Optional[nn.Module]:
    if densenet169 is None or not os.path.exists(path):
        return None
    try:
        model = MammographyClassifier(num_classes=2)
        state = torch.load(path, map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        return model
    except Exception:
        return None


def get_mammography_model() -> Optional[nn.Module]:
    global _MAMMO_MODEL, _MAMMO_LOAD_ATTEMPTED
    if not _MAMMO_LOAD_ATTEMPTED:
        _MAMMO_LOAD_ATTEMPTED = True
        _MAMMO_MODEL = load_mammography_model()
    return _MAMMO_MODEL


def mammography_model_status_text() -> str:
    if os.path.exists(MAMMOGRAPHY_MODEL_PATH):
        return (
            "Model file detected (mammography_model.pt). It will be loaded only "
            "when you run a mammography analysis."
        )
    return (
        "No trained mammography model connected yet — showing a placeholder "
        "analysis only. Train one with train_mammography_kaggle.py on Kaggle, "
        "then add mammography_model.pt next to main.py."
    )


# ---------------------------------------------------------------------------
# 3) PREPROCESSING (must match eval_transform in the training script)
# ---------------------------------------------------------------------------

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def preprocess_mammogram(pil_img: Image.Image) -> torch.Tensor:
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    grayscale_rgb = pil_img.convert("L").convert("RGB")
    return transform(grayscale_rgb).unsqueeze(0)


# ---------------------------------------------------------------------------
# 4) GRAD-CAM (last conv layer of the DenseNet169 backbone)
# ---------------------------------------------------------------------------

def mammography_grad_cam(model: MammographyClassifier, input_tensor: torch.Tensor):
    """Grad-CAM on the last conv feature map of the DenseNet169 backbone.
    Returns (cam in [0,1] as HxW numpy array, reason-if-failed)."""
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

def predict_mammogram(image: np.ndarray):
    """Returns a dict: label, confidence, detail, report_path, is_placeholder,
    explanation_img (PIL image or None)."""
    if image is None:
        return {
            "label": "No image uploaded",
            "confidence": None,
            "detail": "Please upload a mammogram image first.",
            "report_path": None,
            "is_placeholder": True,
            "explanation_img": None,
        }

    try:
        arr = np.asarray(image)
        if arr.size == 0:
            raise ValueError("The uploaded image is empty.")
        if arr.ndim == 2:
            arr = np.repeat(arr[:, :, None], 3, axis=2)
        elif arr.ndim == 3 and arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)
        elif arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[:, :, :3]
        arr = np.asarray(arr, dtype=np.uint8)
        pil = Image.fromarray(arr, mode="RGB")
    except Exception as exc:
        detail = f"The uploaded mammogram could not be processed: {exc}. Please upload a valid image file."
        report_path = None
        fd, report_path = tempfile.mkstemp(suffix=".txt", prefix="mammography_report_")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write("Mammography Analysis Report\n\n")
            f.write(detail)
        return {
            "label": "Analysis error",
            "confidence": None,
            "detail": detail,
            "report_path": report_path,
            "is_placeholder": True,
            "explanation_img": None,
        }

    # Enforce strict gating: use organ detector (if present) and a
    # mammogram-specific heuristic — reject unconditionally if either
    # identifies a non-breast image.
    # Require organ_detector to be present to avoid accidental non-breast analysis.
    if predict_organ is None:
        fd, report_path = tempfile.mkstemp(suffix=".txt", prefix="mammography_report_")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write("Mammography Analysis Report\n\n")
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

    def is_strict_mammogram(img: Image.Image) -> (bool, str):
        gray = np.asarray(img.convert("L"), dtype=np.uint8)
        h, w = gray.shape
        tissue_frac = float((gray > 15).sum()) / (h * w)
        if tissue_frac < 0.02:
            return False, "Too little tissue-like area for a mammogram."
        if tissue_frac > 0.98:
            return False, "Image is almost entirely uniform — unlikely a mammogram."
        # Mammograms may include large black borders or bright presentation
        # windows, so only reject images that are genuinely near-uniform.
        if float(gray.std()) < 6.0:
            return False, "Image has almost no contrast/detail for a mammogram."
        # Accept monochrome scans rendered with a blue tint, but reject images
        # whose channels contain unrelated color information.
        arr_rgb = np.asarray(img.convert("RGB"), dtype=np.float32)
        r, g, b = arr_rgb[..., 0], arr_rgb[..., 1], arr_rgb[..., 2]
        channel_spread = float(np.mean(np.abs(r - g)) + np.mean(np.abs(g - b)) + np.mean(np.abs(r - b)))
        gray_flat = gray.astype(np.float32).reshape(-1)
        channel_correlations = [
            float(np.corrcoef(gray_flat, channel.reshape(-1))[0, 1])
            for channel in (r, g, b)
        ]
        if channel_spread > 14.0 and min(channel_correlations) < 0.85:
            return False, "Image has too much color to be a mammogram."
        return True, ""

    heuristic_ok, heuristic_reason = is_strict_mammogram(pil)

    if organ_label != "mammogram" or organ_conf < ORGAN_DETECTOR_MIN_CONFIDENCE:
        fd, report_path = tempfile.mkstemp(suffix=".txt", prefix="mammography_report_")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write("Mammography Analysis Report\n\n")
            f.write("Rejected: organ detector did not confidently identify a mammogram.\n")
            f.write(f"Detected as: {organ_label} (confidence {organ_conf:.2f})\n")
        return {
            "label": "REJECTED - Not a mammogram",
            "confidence": None,
            "detail": (
                "Organ detector result: "
                f"{organ_label} ({organ_conf * 100:.1f}% confidence). "
                "This image was not analyzed as a mammogram."
            ),
            "report_path": report_path,
            "is_placeholder": True,
            "explanation_img": None,
        }

    if not heuristic_ok:
        fd, report_path = tempfile.mkstemp(suffix=".txt", prefix="mammography_report_")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write("Mammography Analysis Report\n\n")
            f.write("Rejected: uploaded image does not appear to be a mammogram or breast image.\n")
            f.write(f"Reason: {heuristic_reason}\n")
            f.write(f"Organ detector: {organ_label} (confidence {organ_conf:.2f})\n")
        return {
            "label": "REJECTED - Not a mammogram",
            "confidence": None,
            "detail": (
                f"Mammogram image checks failed: {heuristic_reason} "
                f"Organ detector: {organ_label} ({organ_conf * 100:.1f}%)."
            ),
            "report_path": report_path,
            "is_placeholder": True,
            "explanation_img": None,
        }

    try:
        model = get_mammography_model()

        explanation_img = None
        if model is not None:
            input_tensor = preprocess_mammogram(pil)
            with torch.no_grad():
                outputs = model(input_tensor)
                probs = torch.softmax(outputs, dim=1).squeeze(0)
            pred_idx = int(torch.argmax(probs).item())
            confidence = float(probs[pred_idx].item())
            label = CLASS_NAMES[pred_idx]
            detail = (
                f"Model probabilities — BENIGN: {probs[0]*100:.1f}%, "
                f"MALIGNANT: {probs[1]*100:.1f}%. Based on a DenseNet169 model "
                f"fine-tuned on CBIS-DDSM full mammogram images; evaluate this "
                f"against the reported test-set metrics before trusting it. "
                f"Organ detector (advisory): {organ_label} ({organ_conf * 100:.1f}%)."
            )
            is_placeholder = False

            cam, cam_reason = mammography_grad_cam(model, input_tensor)
            if cam is not None:
                explanation_img = gradcam_overlay(pil, cam)
            else:
                print(f"[mammography grad_cam] unavailable: {cam_reason}")
        else:
            label = "Mammography model not yet connected"
            confidence = None
            detail = (
                "This module is ready to display and analyze mammograms, but no "
                "trained mammography model has been connected yet. No prediction "
                "is being made on this image. Run train_mammography_kaggle.py on "
                "Kaggle and add the resulting mammography_model.pt next to "
                "main.py to enable real predictions."
            )
            is_placeholder = True

        report_lines = [
            "Mammography Analysis Report",
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
        fd, report_path = tempfile.mkstemp(suffix=".txt", prefix="mammography_report_")
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
    except Exception as exc:
        detail = f"Mammography analysis hit an unexpected error while processing this image: {exc}. Please try a different image or upload a standard mammogram file."
        fd, report_path = tempfile.mkstemp(suffix=".txt", prefix="mammography_report_")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write("Mammography Analysis Report\n\n")
            f.write(detail)
        return {
            "label": "Analysis error",
            "confidence": None,
            "detail": detail,
            "report_path": report_path,
            "is_placeholder": True,
            "explanation_img": None,
        }


# ---------------------------------------------------------------------------
# 6) RESULT DISPLAY (styled to match the existing app's verdict cards)
# ---------------------------------------------------------------------------

def build_mammography_result_html(result: dict) -> str:
    label = result["label"]
    confidence = result["confidence"]
    detail = result["detail"]

    confidence_html = ""
    if confidence is not None:
        confidence_html = f"<div class='verdict-sub'>Confidence: {confidence * 100:.1f}%</div>"

    color_by_label = {"BENIGN": "#16a34a", "MALIGNANT": "#dc2626"}
    badge_color = color_by_label.get(label, "#334155")

    return f"""
    <div class="verdict-card" style="background:linear-gradient(135deg, {badge_color}, rgba(15,23,42,0.92)); border:1px solid rgba(255,255,255,0.06); box-shadow: 0 18px 36px rgba(2,6,23,0.38), inset 0 1px 0 rgba(255,255,255,0.06);">
      <div class="assist-badge" style="background:rgba(255,255,255,0.08); border-color: rgba(255,255,255,0.12); color:#f8fafc; margin-bottom:12px;">Mammography result</div>
      <div class="verdict-title">{label}</div>
      {confidence_html}
      <div class="verdict-sub" style="margin-top:8px;">{detail}</div>
      <div class="verdict-sub" style="margin-top:10px;font-size:0.8rem;opacity:0.85;">
        AI-generated result for educational/research purposes only.
        This is not a medical diagnosis.
      </div>
    </div>
    """


def analyze_mammogram(image):
    """Top-level function wired directly into the Gradio button click."""
    if image is None:
        html = (
            "<div class='verdict-card' style='background:#334155;'>"
            "<div class='verdict-title'>No image uploaded</div>"
            "<div class='verdict-sub'>Please upload a mammogram image first.</div></div>"
        )
        return html, None, None
    try:
        result = predict_mammogram(image)
        html = build_mammography_result_html(result)
        return html, result["explanation_img"], result["report_path"]
    except Exception as exc:
        detail = f"Mammography analysis unexpectedly failed: {exc}. Please try again with a valid mammogram image."
        html = (
            "<div class='verdict-card' style='background:#334155;'>"
            f"<div class='verdict-title'>Analysis error</div>"
            f"<div class='verdict-sub'>{detail}</div></div>"
        )
        return html, None, None
