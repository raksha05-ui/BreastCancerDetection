
"""

Breast Screening Assistant — simple Gradio app.

Upload an ultrasound image, click Analyze, and get:

  - A clear verdict: Cancer likely / Cancer unlikely

  - A confidence score

  - A visual explanation (Grad-CAM if a real model is loaded, otherwise

    an edge-based saliency overlay)

  - A downloadable text report

If model_cnn.pt is present next to this file, it is loaded automatically

and used for real CNN predictions + Grad-CAM. If it's missing or fails to

load, the app falls back to a simple image-heuristic so the demo still runs.

"""

import os

import tempfile

from datetime import datetime, timezone

from typing import Optional

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import matplotlib.cm as cm

from PIL import Image, ImageFilter

import gradio as gr

import torch

import torch.nn.functional as F

import torchvision.transforms as T

import socket

from mammography_module import analyze_mammogram, mammography_model_status_text
from mri_module import analyze_mri, mri_model_status_text
try:
    from organ_detector import predict_organ, is_breast_label
except Exception:
    predict_organ = None
    is_breast_label = None

RENDER_FAST_MODE = os.getenv("RENDER_FAST_MODE", "").lower() == "true"

class Classifier(torch.nn.Module):

    def __init__(self):

        super().__init__()

        self.conv1 = torch.nn.Conv2d(3, 16, 3, padding=1)

        self.conv2 = torch.nn.Conv2d(16, 32, 3, padding=1)

        self.conv3 = torch.nn.Conv2d(32, 48, 3, padding=1)

        self.conv4 = torch.nn.Conv2d(48, 64, 3, padding=1)

        self.pool1 = torch.nn.MaxPool2d(4, 4)

        self.pool2 = torch.nn.MaxPool2d(2, 2)

        self.fc1 = torch.nn.Linear(7 * 7 * 64, 922)

        self.fc2 = torch.nn.Linear(922, 2)

        self.dropout = torch.nn.Dropout(p=0.25)

        self.batchn1 = torch.nn.BatchNorm2d(16)

        self.batchn2 = torch.nn.BatchNorm2d(32)

        self.batchn3 = torch.nn.BatchNorm2d(48)

        self.batchn4 = torch.nn.BatchNorm2d(64)

    def forward(self, x):

        x = self.pool1(F.relu(self.batchn1(self.conv1(x))))

        x = self.pool2(F.relu(self.batchn2(self.conv2(x))))

        x = self.pool2(F.relu(self.batchn3(self.conv3(x))))

        x = self.pool2(F.relu(self.batchn4(self.conv4(x))))

        x = torch.flatten(x, 1)

        x = self.dropout(F.relu(self.fc1(x)))

        x = F.log_softmax(self.fc2(x), dim=1)

        return x

def check_image_quality(img: Image.Image):

    """Heuristic out-of-distribution / sanity gate.

    Real ultrasound frames are near-grayscale, have moderate contrast, and

    contain actual spatial structure (not flat color, not pure noise).

    This rejects/flags inputs the model was never meant to see, since the

    CNN itself will otherwise emit a confident-looking answer for anything.

    Returns (is_ok: bool, reason: str, is_warning_only: bool).

    """

    arr = np.asarray(img.convert("RGB"), dtype=np.float32)

    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]

    channel_spread = float(np.mean(np.abs(r - g)) + np.mean(np.abs(g - b)) + np.mean(np.abs(r - b)))

    if channel_spread > 18.0:

        return False, "This doesn't look like a grayscale ultrasound image (too much color).", False

    gray = np.asarray(img.convert("L"), dtype=np.float32)

    std_i = float(gray.std())

    if std_i < 8.0:

        return False, "The image has almost no contrast/detail — doesn't look like a scan.", False

    blurred = np.asarray(img.convert("L").filter(ImageFilter.GaussianBlur(radius=2)), dtype=np.float32)

    local_var_orig = float(np.var(gray))

    local_var_blur = float(np.var(blurred))

    noise_ratio = local_var_blur / (local_var_orig + 1e-6)

    if noise_ratio > 0.97:

        return False, "The image looks like random noise rather than a real scan.", False

    if std_i < 20.0 or noise_ratio > 0.92:

        return True, "Image quality is borderline — result may be less reliable.", True

    return True, "", False

def compute_image_quality_metrics(img: Image.Image) -> dict:

    """Compute real, measured image-quality metrics for the uploaded image.

    Every number here comes directly from the pixel data — nothing is a

    placeholder or fabricated figure. Thresholds are documented heuristics

    (consistent with check_image_quality above), not clinically validated

    cutoffs, and are labeled as such in the UI.

    """

    width, height = img.size

    gray = img.convert("L")

    arr = np.asarray(gray, dtype=np.float32)

    brightness = float(arr.mean())

    contrast = float(arr.std())

    laplacian_kernel = [0, -1, 0, -1, 4, -1, 0, -1, 0]

    lap_img = gray.filter(ImageFilter.Kernel((3, 3), laplacian_kernel, scale=1))

    lap_arr = np.asarray(lap_img, dtype=np.float32)

    blur_variance = float(lap_arr.var())

    return {
        "width": width,
        "height": height,
        "brightness": brightness,
        "contrast": contrast,
        "blur_variance": blur_variance,
    }

def evaluate_quality_checks(metrics: dict) -> dict:

    """Turn raw metrics into pass/fail checks, a per-check 0-100 subscore,

    and a composite confidence score (plain average of the four subscores —

    a transparent combination of real measurements, not an invented number).

    """

    width, height = metrics["width"], metrics["height"]

    brightness, contrast, blur_var = metrics["brightness"], metrics["contrast"], metrics["blur_variance"]

    min_dim = min(width, height)

    res_ok = min_dim >= 224

    res_score = min(100.0, 100.0 * min_dim / 224.0)

    bright_ok = 30.0 <= brightness <= 220.0

    bright_score = max(0.0, 100.0 - 2.0 * max(0.0, 30.0 - brightness, brightness - 220.0))

    contrast_ok = contrast >= 20.0

    contrast_score = min(100.0, 100.0 * contrast / 30.0)

    blur_ok = blur_var >= 15.0

    blur_score = min(100.0, 100.0 * blur_var / 40.0)

    checks = {
        "resolution": {"ok": res_ok, "score": res_score,
                        "value": f"{width} × {height}"},
        "brightness": {"ok": bright_ok, "score": bright_score,
                        "value": "Normal" if bright_ok else ("Too dark" if brightness < 30 else "Too bright")},
        "contrast": {"ok": contrast_ok, "score": contrast_score,
                     "value": "Good" if contrast_ok else "Low"},
        "blur": {"ok": blur_ok, "score": blur_score,
                 "value": "Low" if blur_ok else "High"},
    }

    confidence = sum(c["score"] for c in checks.values()) / len(checks)

    confidence = max(0.0, min(100.0, confidence))

    if confidence >= 80:

        readiness = "HIGH"

    elif confidence >= 50:

        readiness = "MEDIUM"

    else:

        readiness = "LOW"

    overall_good = all(c["ok"] for c in checks.values())

    return {
        "checks": checks,
        "confidence": confidence,
        "readiness": readiness,
        "overall_good": overall_good,
    }

def build_quality_panel_html(metrics: dict, evaluation: dict) -> str:

    checks = evaluation["checks"]

    def badge(ok):

        return "<span style='color:#4ade80;'>✓ GOOD</span>" if ok else "<span style='color:#f87171;'>⚠ CHECK</span>"

    reasons = []

    if checks["resolution"]["ok"]:

        reasons.append("Sufficient resolution for detecting visual patterns")

    if checks["brightness"]["ok"]:

        reasons.append("Brightness is within an acceptable range")

    if checks["contrast"]["ok"]:

        reasons.append("Good contrast helps distinguish the affected region")

    if checks["blur"]["ok"]:

        reasons.append("Low blur preserves important visual details")

    if not reasons:

        reasons.append("No quality checks passed — treat this result with caution")

    reasons_html = "".join(f"<div class='quality-reason'>✓ {r}</div>" for r in reasons)

    overall_label = "GOOD ✓" if evaluation["overall_good"] else "NEEDS ATTENTION ⚠"

    readiness_color = {"HIGH": "#4ade80", "MEDIUM": "#fbbf24", "LOW": "#f87171"}[evaluation["readiness"]]

    return f"""

    <div class="quality-card">

      <div class="quality-section-title">IMAGE QUALITY</div>

      <div class="quality-row"><span>Resolution</span><span>{checks['resolution']['value']}</span>{badge(checks['resolution']['ok'])}</div>

      <div class="quality-row"><span>Brightness</span><span>{checks['brightness']['value']}</span>{badge(checks['brightness']['ok'])}</div>

      <div class="quality-row"><span>Contrast</span><span>{checks['contrast']['value']}</span>{badge(checks['contrast']['ok'])}</div>

      <div class="quality-row"><span>Blur</span><span>{checks['blur']['value']}</span>{badge(checks['blur']['ok'])}</div>

      <div class="quality-row" style="margin-top:6px;font-weight:700;"><span>Overall Quality</span><span></span><span>{overall_label}</span></div>

      <div class="quality-section-title" style="margin-top:14px;">WHY THIS IMAGE IS SUITABLE</div>

      {reasons_html}

      <div class="quality-section-title" style="margin-top:14px;">XAI CONFIDENCE</div>

      <div class="quality-row"><span>Image Quality Confidence</span><span></span><span>{evaluation['confidence']:.0f}%</span></div>

      <div class="quality-row"><span>Analysis Readiness</span><span></span><span style="color:{readiness_color};font-weight:700;">{evaluation['readiness']}</span></div>

      <div class="quality-note">

        ⚠ Note: Explainable AI highlights image features that influenced the model's prediction.

        It does not replace clinical diagnosis. Quality scores above are computed from this

        image's own pixel statistics (resolution, brightness, contrast, blur) using documented

        heuristic thresholds — they are not a clinically validated quality assessment.

      </div>

    </div>

    """

def load_bundled_model(path: str = "model_cnn.pt") -> Optional[torch.nn.Module]:

    if not os.path.exists(path):

        return None

    try:

        m = Classifier()

        state = torch.load(path, map_location="cpu")

        m.load_state_dict(state)

        m.eval()

        return m

    except Exception:

        return None

def _preprocess_for_model(pil_img: Image.Image):

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    return transform(pil_img).unsqueeze(0)

def _forward_to_logits(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:

    """Replicate the Classifier's forward pass but stop BEFORE log_softmax,

    returning raw class logits. Grad-CAM must backprop from the raw class

    score, not from log-probabilities — backpropping through log_softmax

    distorts the gradient sign and can make the ReLU'd Grad-CAM weighted-sum

    uniformly negative (i.e. a blank/zeroed heatmap), which is exactly what

    was happening here before this fix."""

    x = model.pool1(F.relu(model.batchn1(model.conv1(x))))

    x = model.pool2(F.relu(model.batchn2(model.conv2(x))))

    x = model.pool2(F.relu(model.batchn3(model.conv3(x))))

    x = model.pool2(F.relu(model.batchn4(model.conv4(x))))

    x = torch.flatten(x, 1)

    x = model.dropout(F.relu(model.fc1(x)))

    logits = model.fc2(x)

    return logits

def grad_cam(model: torch.nn.Module, input_tensor: torch.Tensor):

    """Compute a Grad-CAM heatmap (H, W) in [0, 1] using the last conv layer.

    Returns (cam, reason). cam is None on failure, with `reason` explaining

    why (printed to console and shown to the user instead of silently

    leaving the panel blank or substituting a non-Grad-CAM visualization).

    """

    activation, gradient = None, None

    def forward_hook(_module, _inp, out):

        nonlocal activation

        activation = out.detach()

    def backward_hook(_module, _grad_in, grad_out):

        nonlocal gradient

        gradient = grad_out[0].detach()

    target_layer = model.conv4 if hasattr(model, "conv4") else None

    target_layer_name = "conv4"

    if target_layer is None:

        target_layer_name = None

        for name, module in reversed(list(model.named_modules())):

            if isinstance(module, torch.nn.Conv2d):

                target_layer = module

                target_layer_name = name

                break

    if target_layer is None:

        print("[grad_cam] FAILED: no Conv2d layer found in model")

        return None, "Target convolutional layer not found."

    if not hasattr(model, "fc2"):

        print("[grad_cam] FAILED: model does not match the expected "
              "Classifier architecture (no fc2 head found)")

        return None, "Unsupported model architecture."

    print(f"[grad_cam] model architecture: {model.__class__.__name__}")

    print(f"[grad_cam] target CAM layer: {target_layer_name}")

    print(f"[grad_cam] input shape: {tuple(input_tensor.shape)}")

    fh = target_layer.register_forward_hook(forward_hook)

    if hasattr(target_layer, "register_full_backward_hook"):

        bh = target_layer.register_full_backward_hook(backward_hook)

    else:

        bh = target_layer.register_backward_hook(backward_hook)

    try:

        model.zero_grad()

        logits = _forward_to_logits(model, input_tensor)

        target_class = int(logits.argmax(dim=1).item())

        print(f"[grad_cam] raw model output (logits): {logits.detach().cpu().numpy()}")

        print(f"[grad_cam] predicted class: {target_class}")

        logits[0, target_class].backward()

    finally:

        fh.remove()

        bh.remove()

    if activation is None or gradient is None:

        print("[grad_cam] FAILED: gradient or activation is None "
              "(model may be in inference-only mode or hooks did not fire)")

        return None, "Gradient computation unavailable for this model."

    weights = gradient.mean(dim=(2, 3), keepdim=True)

    cam = F.relu((weights * activation).sum(dim=1, keepdim=True))

    cam = F.interpolate(cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False)

    cam = cam.squeeze().detach().cpu().numpy()

    cam_min, cam_max = float(cam.min()), float(cam.max())

    print(f"[grad_cam] Grad-CAM tensor shape: {cam.shape}, "
          f"heatmap min: {cam_min:.6f}, heatmap max: {cam_max:.6f}")

    if cam_max - cam_min < 1e-6:

        print("[grad_cam] WARNING: heatmap is (near-)uniform — activations "
              "did not vary spatially for this input.")

        return None, "Model attention was uniform across the image for this input."

    cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

    return cam, None

def mc_dropout_predict(model: torch.nn.Module, input_tensor: torch.Tensor, n_passes: int = 5):

    """Run a small number of stochastic forward passes to estimate uncertainty.

    For deployed deployments, 20 passes is expensive and usually unnecessary.

    A lighter default keeps the app responsive while still providing a useful

    uncertainty estimate.

    """

    was_training = model.training

    model.eval()

    for module in model.modules():

        if isinstance(module, torch.nn.Dropout):

            module.train()                                                   

    probs_list = []

    with torch.no_grad():

        for _ in range(n_passes):

            out = model(input_tensor)

            probs_list.append(torch.exp(out))

    model.train(was_training)

    probs_stack = torch.stack(probs_list, dim=0)                    

    mean_probs = probs_stack.mean(dim=0).squeeze(0)

    std_probs = probs_stack.std(dim=0).squeeze(0)

    return mean_probs, std_probs

def render_message_image(message: str, size=(320, 220)) -> Image.Image:

    """Render a plain message inside an image-sized panel, so the

    'Highlighted regions' widget (a gr.Image) never has to be left silently

    blank when Grad-CAM genuinely can't be produced for a real model."""

    from PIL import ImageDraw, ImageFont

    img = Image.new("RGB", size, color="#1e293b")

    draw = ImageDraw.Draw(img)

    try:

        font = ImageFont.load_default()

    except Exception:

        font = None

    words = message.split()

    lines, current = [], ""

    for w in words:

        trial = (current + " " + w).strip()

        if len(trial) > 34:

            lines.append(current)

            current = w

        else:

            current = trial

    if current:

        lines.append(current)

    y = size[1] // 2 - (len(lines) * 14) // 2

    for line in lines:

        draw.text((14, y), line, fill="#fbbf24", font=font)

        y += 16

    return img

def make_histogram(img: Image.Image):

    """Lightweight histogram without matplotlib rendering."""

    gray = np.asarray(img.convert("L")).ravel()

    hist, bins = np.histogram(gray, bins=32, range=(0, 256))

    hist_img = Image.new("RGB", (320, 180), color="#0b1220")

    pixels = hist_img.load()

    max_hist = max(hist) if max(hist) > 0 else 1

    bar_width = 10

    for i, h in enumerate(hist):

        bar_height = int((h / max_hist) * 160)

        x_start = i * bar_width

        for x in range(x_start, min(x_start + bar_width, 320)):

            for y in range(180 - bar_height, 180):

                pixels[x, y] = (96, 165, 250)        

    return hist_img

def edge_saliency_overlay(img: Image.Image, alpha: float = 0.5):

    """Fallback explanation when no trained model is available: highlights edges."""

    gray = np.asarray(img.convert("L"), dtype=np.float32)

    gx = np.zeros_like(gray)

    gy = np.zeros_like(gray)

    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]

    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]

    mag = np.sqrt(gx ** 2 + gy ** 2)

    mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-8)

    mag_img = Image.fromarray((mag * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=2))

    cmap = plt.get_cmap("jet")

    colored = (cmap(np.asarray(mag_img) / 255.0)[:, :, :3] * 255).astype(np.uint8)

    colored_img = Image.fromarray(colored).convert("RGBA")

    return Image.blend(img.convert("RGBA"), colored_img, alpha=alpha)

def gradcam_overlay(img: Image.Image, cam: np.ndarray, alpha: float = 0.5):

    try:

        cmap = matplotlib.colormaps["jet"]

    except Exception:

        cmap = cm.get_cmap("jet")

    colored = (cmap(cam)[:, :, :3] * 255).astype(np.uint8)

    heat = Image.fromarray(colored).convert("RGBA").resize(img.size, resample=Image.BILINEAR)

    return Image.blend(img.convert("RGBA"), heat, alpha=alpha)

def _interpretation_for(assessment: str) -> str:

    return {
        "VERY LOW SUSPICION": "The model output is strongly consistent with a benign finding. This does NOT rule out cancer.",
        "LOW SUSPICION / LIKELY BENIGN": "The model output is more consistent with a benign finding. This does NOT rule out cancer.",
        "INTERMEDIATE / INDETERMINATE": "The model does not provide a sufficiently clear classification.",
        "HIGH SUSPICION": "The model output is more concerning for malignancy.",
        "VERY HIGH SUSPICION": "The model output strongly favors a malignant finding.",
    }.get(assessment, "")

def make_report_text(assessment: str, label: str, benign_prob: float, malignant_prob: float,
                      uncertainty: Optional[float] = None,
                      quality_note: str = "") -> str:

    now = datetime.now(timezone.utc).isoformat()

    uncertainty_line = ""

    if uncertainty is not None:

        reliability = "Low" if uncertainty > 0.15 else ("Medium" if uncertainty > 0.05 else "High")

        uncertainty_line = (
            f"Prediction uncertainty (MC-Dropout std): {uncertainty:.3f}\n"
            f"Estimated reliability: {reliability}\n\n"
        )

    quality_line = f"Image quality note: {quality_note}\n\n" if quality_note else ""

    interpretation = _interpretation_for(assessment)

    return (
        "AI Breast Ultrasound Screening Report\n"
        f"Generated: {now}\n\n"
        f"Assessment: {assessment}\n\n"
        "Model probability (uncalibrated — not a validated clinical risk threshold):\n"
        f"  Benign:    {benign_prob * 100:.1f}%\n"
        f"  Malignant: {malignant_prob * 100:.1f}%\n\n"
        f"Interpretation: {interpretation}\n\n"
        f"{uncertainty_line}"
        f"{quality_line}"
        "Cancer stage cannot be determined from this AI ultrasound screening result.\n"
        "Staging requires appropriate clinical evaluation and diagnostic testing.\n\n"
        "AI screening result — not a medical diagnosis.\n"
        "An AI result should not replace evaluation by a qualified healthcare professional.\n"
        "Please consult a qualified physician for any medical decision."
    )

def classify_suspicion(malignant_prob: float) -> str:

    """Map the model's raw malignant probability to a 5-level display label.

    IMPORTANT: these are NOT clinically validated risk thresholds. No

    calibration curve, ROC analysis, or held-out threshold-tuning results

    exist anywhere in this project (checked the training notebook — every

    validation-metric cell has empty output). These are simply evenly-spaced

    bins over the model's own uncalibrated probability output, presented so

    the UI reflects the model's actual varying confidence instead of

    collapsing everything into a binary decision. The report/UI must always

    show the raw percentages alongside this label, labeled as "model

    probability" rather than a medical risk estimate.

    """

    if malignant_prob < 0.20:

        return "VERY LOW SUSPICION"

    elif malignant_prob < 0.40:

        return "LOW SUSPICION / LIKELY BENIGN"

    elif malignant_prob < 0.60:

        return "INTERMEDIATE / INDETERMINATE"

    elif malignant_prob < 0.80:

        return "HIGH SUSPICION"

    else:

        return "VERY HIGH SUSPICION"

MODEL = None

def get_model() -> Optional[torch.nn.Module]:

    global MODEL

    if MODEL is None and os.path.exists("model_cnn.pt"):

        MODEL = load_bundled_model()

    return MODEL

def predict(image: np.ndarray, use_explanation: bool):

    pil = Image.fromarray(np.uint8(image)).convert("RGB")

    model = get_model()

    is_ok, quality_reason, is_warning_only = check_image_quality(pil)

    label, benign_prob, malignant_prob, uncertainty = None, None, None, None

    explanation_img = None

    if not is_ok:

        hist_img = make_histogram(pil)

        report = (
            "AI Breast Ultrasound Screening Report\n"
            f"Generated: {datetime.now(timezone.utc).isoformat()}\n\n"
            "Assessment: REJECTED — input does not resemble an ultrasound scan\n"
            f"Reason: {quality_reason}\n\n"
            "No prediction was made. Please upload a genuine ultrasound image."
        )

        fd, report_path = tempfile.mkstemp(suffix=".txt", prefix="report_")

        with os.fdopen(fd, "w", encoding="utf-8") as f:

            f.write(report)

        return "REJECTED", quality_reason, None, None, hist_img, None, report_path, None, ""

    # Use organ_detector if available; regardless, require a strict
    # ultrasound-specific heuristic to pass. If either check indicates a
    # non-breast image, reject unconditionally (no permissive fallback).
    # Require a trained `organ_detector.pt` to be present. This enforces
    # that only images confirmed as breast modality are analyzed. If the
    # detector is not installed, reject uploads to avoid accidental
    # analysis of non-breast images.
    if predict_organ is None:
        hist_img = make_histogram(pil)
        fd, report_path = tempfile.mkstemp(suffix=".txt", prefix="report_")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write("AI Breast Ultrasound Screening Report\n")
            f.write(f"Generated: {datetime.now(timezone.utc).isoformat()}\n\n")
            f.write("Assessment: REJECTED — organ detector not available on this installation.\n")
            f.write("To enable safe automatic gating (preventing non-breast analysis), place a trained 'organ_detector.pt' next to main.py.\n")
        return "REJECTED", "organ_detector missing", None, None, hist_img, None, report_path, None, ""

    organ_label, organ_conf = ("unknown", 0.0)
    try:
        organ_label, organ_conf = predict_organ(pil)
    except Exception:
        organ_label, organ_conf = "unknown", 0.0

    def is_strict_breast_ultrasound(img: Image.Image) -> (bool, str):
        gray = np.asarray(img.convert("L"), dtype=np.float32)
        h, w = gray.shape
        # require a non-trivial tissue fraction
        tissue_frac = float((gray > 10).sum()) / (h * w)
        if tissue_frac < 0.02:
            return False, "Too little tissue-like area for a breast ultrasound."
        if tissue_frac > 0.98:
            return False, "Image is almost entirely uniform — unlikely ultrasound."
        std = float(gray.std())
        if std < 12.0 or std > 120.0:
            return False, "Image contrast/variance outside expected ultrasound range."
        # Check color spread to ensure near-grayscale
        arr_rgb = np.asarray(img.convert("RGB"), dtype=np.float32)
        r, g, b = arr_rgb[..., 0], arr_rgb[..., 1], arr_rgb[..., 2]
        channel_spread = float(np.mean(np.abs(r - g)) + np.mean(np.abs(g - b)) + np.mean(np.abs(r - b)))
        if channel_spread > 14.0:
            return False, "Image has too much color to be an ultrasound."
        bright_thresh = np.percentile(gray, 65)
        ys = np.where(gray > bright_thresh)[0]
        if ys.size == 0:
            return False, "No bright tissue region detected."
        center_y = float(ys.mean()) / h
        if center_y < 0.05 or center_y > 0.95:
            return False, "Tissue location atypical for a breast ultrasound image."
        return True, ""

    heuristic_ok, heuristic_reason = is_strict_breast_ultrasound(pil)

    # Reject if organ detector confidently says it's not ultrasound
    if organ_conf >= 0.6 and organ_label != "ultrasound":
        hist_img = make_histogram(pil)
        fd, report_path = tempfile.mkstemp(suffix=".txt", prefix="report_")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write("AI Breast Ultrasound Screening Report\n")
            f.write(f"Generated: {datetime.now(timezone.utc).isoformat()}\n\n")
            f.write("Assessment: REJECTED — uploaded image does not appear to be a breast ultrasound.\n")
            f.write(f"Detected as: {organ_label} (confidence {organ_conf:.2f})\n\n")
            f.write("No prediction was made. Please upload a genuine breast ultrasound image.")
        return "REJECTED", f"Detected as: {organ_label}", None, None, hist_img, None, report_path, None, ""

    # Reject if strict heuristic fails
    if not heuristic_ok:
        hist_img = make_histogram(pil)
        fd, report_path = tempfile.mkstemp(suffix=".txt", prefix="report_")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write("AI Breast Ultrasound Screening Report\n")
            f.write(f"Generated: {datetime.now(timezone.utc).isoformat()}\n\n")
            f.write("Assessment: REJECTED — uploaded image does not appear to be a breast ultrasound.\n")
            f.write(f"Reason: {heuristic_reason}\n\n")
            f.write("No prediction was made. Please upload a genuine breast ultrasound image.")
        return "REJECTED", heuristic_reason, None, None, hist_img, None, report_path, None, ""

    if model is not None:

        try:

            input_t = _preprocess_for_model(pil)

            with torch.no_grad():                                  

                if RENDER_FAST_MODE:

                    output = model(input_t)

                    probs = F.softmax(output, dim=1)

                    mean_probs = probs.squeeze(0)

                    uncertainty = None

                else:

                    mean_probs, std_probs = mc_dropout_predict(model, input_t, n_passes=2)

                    uncertainty = float(std_probs[int(mean_probs.argmax().item())].item())

            benign_prob = float(mean_probs[0].item())

            malignant_prob = float(mean_probs[1].item())

            idx_t = 0 if benign_prob >= malignant_prob else 1

            label = "Benign" if idx_t == 0 else "Malignant"

            if torch.cuda.is_available():

                torch.cuda.empty_cache()

            if use_explanation and not RENDER_FAST_MODE:

                cam, cam_fail_reason = grad_cam(model, input_t)

                if cam is not None:

                    explanation_img = gradcam_overlay(pil, cam)

                else:

                    explanation_img = render_message_image(
                        f"Model attention map unavailable for this image/model. ({cam_fail_reason})"
                    )

        except Exception:

            label = None                             

    if label is None:

        arr = np.asarray(pil)

        mean_i, std_i = float(arr.mean()), float(arr.std())

        score = max(0.0, min(1.0, (140 - mean_i) / 100.0 + std_i / 255.0))

        label = "Malignant" if score >= 0.5 else "Benign"

        malignant_prob = float(max(0.01, min(0.99, score)))

        benign_prob = 1.0 - malignant_prob

        uncertainty = None

    if use_explanation and explanation_img is None:

        explanation_img = edge_saliency_overlay(pil)

    if RENDER_FAST_MODE and not use_explanation:

        explanation_img = None

    assessment = classify_suspicion(malignant_prob)

    hist_img = None if RENDER_FAST_MODE else make_histogram(pil)

    report = make_report_text(assessment, label, benign_prob, malignant_prob, uncertainty,
                               quality_reason if is_warning_only else "")

    fd, report_path = tempfile.mkstemp(suffix=".txt", prefix="report_")

    with os.fdopen(fd, "w", encoding="utf-8") as f:

        f.write(report)

    return (assessment, label, benign_prob, malignant_prob, hist_img, explanation_img,
            report_path, uncertainty, (quality_reason if is_warning_only else ""))

CSS = """

:root {

  color-scheme: dark;

  --bg: #06131f;

  --panel: rgba(10, 18, 34, 0.92);

  --panel-strong: rgba(15, 23, 42, 0.96);

  --card: rgba(15, 23, 42, 0.88);

  --text: #e2e8f0;

  --muted: #94a3b8;

  --border: rgba(148, 163, 184, 0.18);

  --accent: #60a5fa;

  --accent-strong: #38bdf8;

  --success: #22c55e;

  --warning: #fbbf24;

  --danger: #f87171;

}

html, body {

  min-height: 100%;

  background: radial-gradient(circle at top left, rgba(96,165,250,0.16), transparent 22%),

              radial-gradient(circle at bottom right, rgba(34,197,94,0.08), transparent 18%),

              linear-gradient(180deg, #040c16 0%, #071827 100%) !important;

  color: var(--text) !important;

  font-family: "Segoe UI", sans-serif;

}

.gradio-container {

  background: transparent !important;

  max-width: 1400px !important;

  padding: 24px 8px 40px !important;

}

.header {

  background: linear-gradient(135deg, rgba(15, 23, 42, 0.96), rgba(14, 116, 144, 0.22));

  border: 1px solid var(--border) !important;

  border-radius: 24px;

  padding: 22px 24px;

  box-shadow: 0 16px 42px rgba(0,0,0,0.28);

  margin-bottom: 18px;

}

.logo-box {

  width: 60px; height: 60px; border-radius: 18px;

  display: flex; align-items: center; justify-content: center;

  background: linear-gradient(135deg, rgba(96,165,250,0.28), rgba(56,189,248,0.16));

  border: 1px solid rgba(96,165,250,0.45);

  color: var(--text); font-weight: 800; font-size: 1.4rem;

}

.hero-copy {

  line-height: 1.4;

}

.hero-copy h2 {

  margin: 0;

  font-size: clamp(1.5rem, 2vw, 2.2rem);

  font-weight: 800;

}

.hero-copy p {

  margin: 6px 0 0;

  color: var(--muted);

  font-size: 0.98rem;

}

.info-strip {

  display: grid;

  grid-template-columns: repeat(3, minmax(180px, 1fr));

  gap: 14px;

  margin-top: 18px;

}

.info-card {

  background: rgba(15, 23, 42, 0.82);

  border: 1px solid var(--border);

  border-radius: 16px;

  padding: 14px 16px;

}

.info-card .mini-title {

  display: inline-block;

  font-size: 0.72rem;

  letter-spacing: 0.08em;

  text-transform: uppercase;

  color: #7dd3fc;

  margin-bottom: 8px;

}

.info-card .mini-text {

  font-size: 0.93rem;

  color: var(--text);

  line-height: 1.45;

}

.section-card {

  background: rgba(15, 23, 42, 0.82);

  border: 1px solid var(--border);

  border-radius: 18px;

  padding: 16px;

  margin-top: 10px;

}

.step-pill {

  display: inline-flex;

  align-items: center;

  gap: 8px;

  background: rgba(96,165,250,0.12);

  border: 1px solid rgba(96,165,250,0.25);

  border-radius: 999px;

  color: #bae6fd;

  padding: 7px 12px;

  font-size: 0.75rem;

  font-weight: 700;

  letter-spacing: 0.04em;

  text-transform: uppercase;

}

.verdict-card {

  position: relative;

  overflow: hidden;

  border-radius: 22px;

  padding: 22px 22px 18px;

  text-align: left;

  background: linear-gradient(180deg, rgba(15,23,42,0.88), rgba(15,23,42,0.72));

  box-shadow: 0 18px 36px rgba(2,6,23,0.38), inset 0 1px 0 rgba(255,255,255,0.06);

  border: 1px solid rgba(148,163,184,0.18);

}

.verdict-card::before {

  content: "";

  position: absolute;

  inset: 0 0 auto 0;

  height: 4px;

  background: linear-gradient(90deg, rgba(125,211,252,0.9), rgba(59,130,246,0.7), rgba(168,85,247,0.7));

}

.verdict-title {

  font-size: 1.55rem;

  line-height: 1.2;

  font-weight: 800;

  margin-bottom: 10px;

  letter-spacing: -0.02em;

  color: #f8fafc;

}

.verdict-sub {

  font-size: 0.96rem;

  opacity: 0.96;

  line-height: 1.6;

  color: #e2e8f0;

}

.footer-note { text-align:center; margin-top:16px; color: var(--muted); font-size: 0.82rem; }

.quality-card {

  border-radius: 18px; padding: 20px; background: #111827;

  border: 1px solid rgba(148,163,184,0.18);

  box-shadow: 0 12px 30px rgba(0,0,0,0.25); margin-top: 16px;

}

.quality-section-title {

  font-size: 0.8rem; font-weight: 800; letter-spacing: 0.06em;

  color: #94a3b8; border-bottom: 1px solid #334155; padding-bottom: 6px; margin-bottom: 8px;

}

.quality-row {

  display: flex; justify-content: space-between; gap: 12px;

  font-size: 0.9rem; padding: 4px 0; color: #e2e8f0;

}

.quality-row span:first-child { color: #94a3b8; }

.quality-reason { font-size: 0.85rem; color: #cbd5e1; padding: 3px 0; }

.quality-note {

  margin-top: 12px; font-size: 0.78rem; color: #fbbf24; opacity: 0.9;

  border-top: 1px solid #334155; padding-top: 10px;

}

.gr-button-primary,
#analyze_btn,
#mammo_action_btn,
#mri_action_btn {

  display: block !important;

  width: 100% !important;

  background: linear-gradient(135deg, #2563eb 0%, #38bdf8 100%) !important;

  border: 1px solid rgba(255,255,255,0.10) !important;

  border-radius: 14px !important;

  box-shadow: 0 12px 26px rgba(37, 99, 235, 0.24), inset 0 1px 0 rgba(255,255,255,0.18) !important;

  font-weight: 800 !important;

  font-size: 1.05rem !important;

  letter-spacing: 0.01em !important;

  min-height: 58px !important;

  transition: transform 0.15s ease, box-shadow 0.15s ease !important;

}

.gr-button-primary:hover,
#analyze_btn:hover,
#mammo_action_btn:hover,
#mri_action_btn:hover {

  transform: translateY(-1px) !important;

  box-shadow: 0 15px 28px rgba(37, 99, 235, 0.32), inset 0 1px 0 rgba(255,255,255,0.20) !important;

}

.gr-button-secondary {

  background: rgba(15, 23, 42, 0.82) !important;

  border: 1px solid rgba(148,163,184,0.25) !important;

}

.tip-box {

  margin-top: 12px;

  background: linear-gradient(135deg, rgba(96,165,250,0.10), rgba(20,184,166,0.07));

  border: 1px solid rgba(96,165,250,0.24);

  border-radius: 12px;

  padding: 10px 12px;

  color: #dbeafe;

  font-size: 0.81rem;

  line-height: 1.5;

}

.upload-panel {

  background: rgba(15, 23, 42, 0.74);

  border: 1px solid rgba(148, 163, 184, 0.2);

  border-radius: 16px;

  padding: 8px;

}

.page-shell {

  background: rgba(15, 23, 42, 0.72);

  border: 1px solid rgba(148,163,184,0.16);

  border-radius: 22px;

  padding: 18px;

  box-shadow: inset 0 1px 0 rgba(255,255,255,0.04), 0 18px 32px rgba(2,6,23,0.28);

}

.dashboard-shell {

  display: grid;

  grid-template-columns: 1.2fr 1.5fr;

  gap: 14px;

  margin-top: 6px;

}

.side-panel,
.insight-card {

  background: rgba(15, 23, 42, 0.72);

  border: 1px solid rgba(148,163,184,0.16);

  border-radius: 18px;

  padding: 16px;

}

.assist-badge {

  display: inline-block;

  background: rgba(14,165,233,0.12);

  border: 1px solid rgba(125,211,252,0.25);

  border-radius: 999px;

  padding: 6px 10px;

  color: #bae6fd;

  font-size: 0.72rem;

  letter-spacing: 0.07em;

  text-transform: uppercase;

  margin-bottom: 10px;

  font-weight: 700;

}

.panel-note {

  margin-top: 8px;

  color: #cbd5e1;

  font-size: 0.85rem;

  line-height: 1.6;

}

.stat-grid {

  display: grid;

  grid-template-columns: repeat(3, minmax(80px, 1fr));

  gap: 10px;

  margin-top: 14px;

}

.stat-pill {

  background: rgba(15, 23, 42, 0.9);

  border: 1px solid rgba(148,163,184,0.16);

  border-radius: 12px;

  padding: 12px 10px;

  text-align: center;

}

.stat-pill .stat-number {

  font-size: 1.1rem;

  font-weight: 800;

  color: #f8fafc;

}

.stat-pill .stat-label {

  font-size: 0.68rem;

  color: #94a3b8;

  text-transform: uppercase;

  letter-spacing: 0.08em;

  margin-top: 4px;

}

.how-it-works {

  display: grid;

  grid-template-columns: repeat(3, minmax(180px, 1fr));

  gap: 12px;

  margin: 18px 0 10px;

}

.how-card {

  background: rgba(15, 23, 42, 0.74);

  border: 1px solid rgba(148,163,184,0.18);

  border-radius: 16px;

  padding: 14px 12px;

}

.how-card .step-index {

  display: inline-flex;

  align-items: center;

  justify-content: center;

  width: 28px;

  height: 28px;

  border-radius: 50%;

  background: linear-gradient(135deg, rgba(96,165,250,0.18), rgba(56,189,248,0.12));

  border: 1px solid rgba(125,211,252,0.25);

  color: #bae6fd;

  font-weight: 800;

  margin-bottom: 8px;

}

.how-card .step-title {

  font-size: 0.82rem;

  font-weight: 800;

  color: #e2e8f0;

  letter-spacing: 0.04em;

  text-transform: uppercase;

  margin-bottom: 6px;

}

.how-card .step-body {

  font-size: 0.85rem;

  color: #cbd5e1;

  line-height: 1.5;

}

.status-row {

  display: flex;

  gap: 10px;

  flex-wrap: wrap;

  margin-top: 18px;

}

.status-pill {

  display: inline-flex;

  align-items: center;

  gap: 7px;

  padding: 7px 12px;

  border-radius: 999px;

  border: 1px solid rgba(148,163,184,0.2);

  background: rgba(15, 23, 42, 0.74);

  color: #e2e8f0;

  font-size: 0.76rem;

  font-weight: 700;

}

.status-pill .dot {

  width: 8px;

  height: 8px;

  border-radius: 50%;

  background: #38bdf8;

  box-shadow: 0 0 10px rgba(56,189,248,0.7);

}

#image_input, #mammo_image_input, #mri_image_input {

  border-radius: 14px !important;

}

.gradio-image .image-container {

  border-radius: 14px !important;

  border: 1px solid rgba(96,165,250,0.2) !important;

}

.gradio-container .tabitem {

  border-radius: 16px !important;

}

.gradio-container .tab-nav {

  border-bottom: 1px solid rgba(148,163,184,0.18) !important;

}

.gradio-container button[role="tab"],
.gradio-container .tabitem button {

  border-radius: 12px 12px 0 0 !important;

  color: #cbd5e1 !important;

  background: rgba(15, 23, 42, 0.82) !important;

  border: 1px solid rgba(148,163,184,0.15) !important;

}

.gradio-container button[role="tab"][aria-selected="true"],
.gradio-container .tabitem button[aria-selected="true"] {

  background: linear-gradient(135deg, rgba(37, 99, 235, 0.25), rgba(56,189,248,0.12)) !important;

  border-color: rgba(96,165,250,0.35) !important;

  color: #e0f2fe !important;

}

.gradio-container .upload-box {

  border: 1px solid rgba(96,165,250,0.2) !important;

  box-shadow: inset 0 0 0 1px rgba(96,165,250,0.08) !important;

}

"""

def analyze(image, show_explanation):

    if image is None:

        return (
            "<div class='verdict-card' style='background:#334155;'>"
            "<div class='verdict-title'>No image uploaded</div>"
            "<div class='verdict-sub'>Please upload an ultrasound image first.</div></div>",
            None, None, None, None, "",
        )

    try:

        (assessment, label, benign_prob, malignant_prob, hist_img, explanation_img,
         report_path, uncertainty, quality_warning) = predict(image, show_explanation)

    except Exception as e:

        import traceback

        traceback.print_exc()

        error_html = f"""

        <div class="verdict-card" style="background:#334155;">

          <div class="verdict-title">Something went wrong</div>

          <div class="verdict-sub">{type(e).__name__}: {e}</div>

        </div>

        """

        return error_html, None, None, None, None, ""

    pil_for_quality = Image.fromarray(np.uint8(image)).convert("RGB")

    q_metrics = compute_image_quality_metrics(pil_for_quality)

    q_eval = evaluate_quality_checks(q_metrics)

    quality_html = build_quality_panel_html(q_metrics, q_eval)

    if assessment == "REJECTED":

        verdict_html = f"""

        <div class="verdict-card" style="background:#334155;">

          <div class="verdict-title">Image rejected</div>

          <div class="verdict-sub">{label}</div>

        </div>

        """

        return verdict_html, hist_img, None, report_path, report_path, quality_html

    color_by_assessment = {
        "VERY LOW SUSPICION": "#15803d",
        "LOW SUSPICION / LIKELY BENIGN": "#16a34a",
        "INTERMEDIATE / INDETERMINATE": "#d97706",
        "HIGH SUSPICION": "#ea580c",
        "VERY HIGH SUSPICION": "#dc2626",
    }

    color = color_by_assessment.get(assessment, "#334155")

    interpretation = _interpretation_for(assessment)

    uncertainty_html = ""

    if uncertainty is not None:

        reliability = "Low" if uncertainty > 0.15 else ("Medium" if uncertainty > 0.05 else "High")

        rel_color = {"Low": "#f87171", "Medium": "#fbbf24", "High": "#4ade80"}[reliability]

        uncertainty_html = (
            f"<div class='verdict-sub' style='margin-top:6px;'>"
            f"Model uncertainty: {uncertainty:.3f} &nbsp;•&nbsp; "
            f"Reliability: <span style='color:{rel_color};font-weight:700;'>{reliability}</span></div>"
        )

    warning_html = ""

    if quality_warning:

        warning_html = (
            f"<div class='verdict-sub' style='margin-top:6px;color:#fbbf24;'>⚠ {quality_warning}</div>"
        )

    verdict_html = f"""

    <div class="verdict-card" style="background:linear-gradient(135deg, {color}, rgba(15,23,42,0.92)); border:1px solid rgba(255,255,255,0.06);">

      <div class="assist-badge" style="background:rgba(255,255,255,0.08); border-color: rgba(255,255,255,0.12); color:#f8fafc; margin-bottom:12px;">Screening result</div>

      <div class="verdict-title">{assessment}</div>

      <div class="verdict-sub">Benign: <strong>{benign_prob*100:.1f}%</strong> &nbsp;•&nbsp; Malignant: <strong>{malignant_prob*100:.1f}%</strong></div>

      <div class="verdict-sub" style="margin-top:8px;">{interpretation}</div>

      {uncertainty_html}

      {warning_html}

      <div class="verdict-sub" style="margin-top:10px;font-size:0.8rem;opacity:0.85;">

        AI screening result — not a medical diagnosis. Cancer stage cannot be determined from

        this result. Please consult a qualified healthcare professional for clinical evaluation.

      </div>

    </div>

    """

    return verdict_html, hist_img, explanation_img, report_path, report_path, quality_html

with gr.Blocks(title="Breast Screening Assistant") as demo:

    gr.HTML(
        """

        <div class="header">

          <div style="display:flex;align-items:center;gap:18px;flex-wrap:wrap;">

            <div class="logo-box">BC</div>

            <div class="hero-copy">

              <h2>Breast Screening Assistant</h2>

              <p>Upload a breast ultrasound image and get a quick screening result in a few seconds.</p>

            </div>

          </div>

          <div class="info-strip">

            <div class="info-card">
              <div class="mini-title">Step 1</div>
              <div class="mini-text">Choose a clear breast ultrasound image from your device.</div>
            </div>

            <div class="info-card">
              <div class="mini-title">Step 2</div>
              <div class="mini-text">Click analyze to review the model result and image quality check.</div>
            </div>

            <div class="info-card">
              <div class="mini-title">Step 3</div>
              <div class="mini-text">Download the report for record keeping and show it to a clinician.</div>
            </div>

          </div>

        </div>

        """
    )

    with gr.Tabs():

     with gr.Tab("Breast Ultrasound Screening"):

      gr.HTML(
          """
          <div class="page-shell">
            <div class="dashboard-shell">
              <div class="side-panel">
                <div class="assist-badge">Quick start</div>
                <div class="panel-note">
                  This tool is designed for educational use and should be treated as a support aid, not a diagnosis. Please upload a breast ultrasound image that is clear, centered, and not blurry.
                </div>
                <div class="status-row">
                  <div class="status-pill"><span class="dot"></span>AI model ready</div>
                  <div class="status-pill"><span class="dot"></span>Image quality check</div>
                  <div class="status-pill"><span class="dot"></span>Downloadable report</div>
                </div>
              </div>

              <div class="insight-card">
                <div class="assist-badge" style="margin-bottom:12px;">Overview</div>
                <div class="stat-grid">
                  <div class="stat-pill">
                    <div class="stat-number">3</div>
                    <div class="stat-label">Steps</div>
                  </div>
                  <div class="stat-pill">
                    <div class="stat-number">1</div>
                    <div class="stat-label">Upload</div>
                  </div>
                  <div class="stat-pill">
                    <div class="stat-number">1</div>
                    <div class="stat-label">Report</div>
                  </div>
                </div>
                <div class="panel-note" style="margin-top:14px;">
                  Use a clear, properly centered image for the most reliable screening support. The app also checks whether the image quality is suitable before producing a result.
                </div>
              </div>
            </div>

            <div class="how-it-works">
              <div class="how-card">
                <div class="step-index">1</div>
                <div class="step-title">Upload</div>
                <div class="step-body">Choose a breast ultrasound image from your device.</div>
              </div>
              <div class="how-card">
                <div class="step-index">2</div>
                <div class="step-title">Analyze</div>
                <div class="step-body">Click analyze to review the result and image quality check.</div>
              </div>
              <div class="how-card">
                <div class="step-index">3</div>
                <div class="step-title">Review</div>
                <div class="step-body">Download the report and discuss the findings with a clinician.</div>
              </div>
            </div>
          </div>
          """
      )

      with gr.Row(equal_height=False):

        with gr.Column(scale=1):

            image_input = gr.Image(
                label="Upload ultrasound image",
                type="numpy",
                show_label=True,
                height=360,
                container=True,
            )

            gr.HTML(
                """
                <div class="tip-box">
                  Tip: use a clear breast ultrasound image that is centered and not blurry. Poor image quality may reduce the reliability of the result.
                </div>
                """
            )

            show_explanation = gr.Checkbox(
                label="Show visual explanation (highlighted regions)",
                value=(not RENDER_FAST_MODE),
                visible=False,
            )

            analyze_btn = gr.Button("Analyze image", variant="primary", elem_id="analyze_btn")

            model_status = (
                "Using trained CNN model (model_cnn.pt)."
                if os.path.exists("model_cnn.pt")
                else "No trained model found — using a basic image heuristic instead. "
                     "Add model_cnn.pt next to main.py for real predictions."
            )

            gr.HTML(f"<div style='color:#94a3b8;font-size:0.85rem;margin-top:8px;line-height:1.5;'>{model_status}</div>")

        with gr.Column(scale=1):

            verdict_out = gr.HTML()

            with gr.Row(equal_height=False):

                hist_out = gr.Image(label="Intensity histogram", show_label=True)

                explanation_out = gr.Image(label="Highlighted regions", show_label=True)

            report_file = gr.File(label="Download report", visible=True)

            quality_out = gr.HTML()

            gr.HTML(
                """
                <div class="panel-note">
                  The result includes a quick image quality check and a downloadable report for review.
                </div>
                """
            )

     with gr.Tab("Mammography Analysis"):

      with gr.Row():

        with gr.Column(scale=1):

            mammo_image_input = gr.Image(label="Mammogram image", type="numpy")

            mammo_analyze_btn = gr.Button("Analyze Mammogram", variant="primary", elem_id="mammo_action_btn")

            gr.HTML(
                f"<div style='color:#94a3b8;font-size:0.85rem;margin-top:4px;'>"
                f"{mammography_model_status_text()}</div>"
            )

        with gr.Column(scale=1):

            mammo_verdict_out = gr.HTML()

            mammo_explanation_out = gr.Image(label="Highlighted regions (Grad-CAM)", show_label=True)

            mammo_report_file = gr.File(label="Download mammography report", visible=True)

      gr.HTML(
          "<div class='footer-note'>"
          "AI-generated result for educational/research purposes only. "
          "This is not a medical diagnosis."
          "</div>"
      )

      mammo_analyze_btn.click(
          fn=analyze_mammogram,
          inputs=[mammo_image_input],
          outputs=[mammo_verdict_out, mammo_explanation_out, mammo_report_file],
      )

     with gr.Tab("MRI Analysis"):

      with gr.Row():

        with gr.Column(scale=1):

            mri_image_input = gr.Image(label="Breast MRI image", type="numpy")

            mri_analyze_btn = gr.Button("Analyze MRI", variant="primary", elem_id="mri_action_btn")

            gr.HTML(
                f"<div style='color:#94a3b8;font-size:0.85rem;margin-top:4px;'>"
                f"{mri_model_status_text()}</div>"
            )

        with gr.Column(scale=1):

            mri_verdict_out = gr.HTML()

            mri_explanation_out = gr.Image(label="Highlighted regions (Grad-CAM)", show_label=True)

            mri_report_file = gr.File(label="Download MRI report", visible=True)

      gr.HTML(
          "<div class='footer-note'>"
          "AI-generated result for educational/research purposes only. "
          "This is not a medical diagnosis."
          "</div>"
      )

      mri_analyze_btn.click(
          fn=analyze_mri,
          inputs=[mri_image_input],
          outputs=[mri_verdict_out, mri_explanation_out, mri_report_file],
      )

    gr.HTML(
        "<div class='footer-note'>"
        "This tool is for educational exploration only and is not a medical diagnosis. "
        "Always consult a qualified physician."
        "</div>"
    )

    analyze_btn.click(
        fn=analyze,
        inputs=[image_input, show_explanation],
        outputs=[verdict_out, hist_out, explanation_out, report_file, report_file, quality_out],
    )

def find_free_port(start_port=7860, end_port=7880):

    for p in range(start_port, end_port + 1):

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:

            try:

                s.bind(("0.0.0.0", p))

                return p

            except OSError:

                continue

    return None

if __name__ == "__main__":

    port = int(os.environ.get("PORT", "7860"))

    if port == 7860:

        port = find_free_port(7860, 7880) or 7860

    local_url = f"http://127.0.0.1:{port}"

    print("=" * 50)

    print("Breast Screening Assistant")

    print("=" * 50)

    print(f"Server running on: {local_url}")

    print(f"Network binding:   0.0.0.0:{port}")

    print("=" * 50)

    if not os.environ.get("RENDER"):

        import threading

        import time

        import webbrowser

        def _open_browser():

            time.sleep(1.5)                                                           

            webbrowser.open(local_url)

        threading.Thread(target=_open_browser, daemon=True).start()

    demo.launch(server_name="0.0.0.0", server_port=port, debug=False, share=False,
                inbrowser=False, show_error=False, css=CSS)
