import cv2
import numpy as np

from app.schemas.project import QualityIssue, QualityReport

MIN_SHORT_SIDE = (480, 720)  # reject below the first, warn below the second
MIN_SHARPNESS = (40.0, 400.0)  # variance of the Laplacian on the normalised image
BRIGHTNESS_RANGE = ((45, 215), (70, 190))  # (reject range), (warn range) of mean luminance
MIN_CONTRAST = 25.0


def _issue(code: str, severity: str, message: str) -> QualityIssue:
    return QualityIssue(code=code, severity=severity, message=message)


def assess(image: np.ndarray, original_size: tuple[int, int]) -> QualityReport:
    """Reject unusable photos and tell the user how to retake them."""
    issues: list[QualityIssue] = []
    short_side = min(original_size)
    if short_side < MIN_SHORT_SIDE[0]:
        issues.append(_issue("low_resolution", "error", f"The photo is too small ({short_side}px on its short side). Use at least {MIN_SHORT_SIDE[0]}px, ideally a full-size photo from your phone."))
    elif short_side < MIN_SHORT_SIDE[1]:
        issues.append(_issue("low_resolution", "warning", "The photo has a low resolution; areas will be less precise. A full-size photo works best."))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if sharpness < MIN_SHARPNESS[0]:
        issues.append(_issue("blurry", "error", "The photo is too blurry. Hold the phone steady, tap to focus on the house and retake it."))
    elif sharpness < MIN_SHARPNESS[1]:
        issues.append(_issue("blurry", "warning", "The photo looks slightly soft. A sharper photo gives cleaner region outlines."))

    mean, contrast = float(gray.mean()), float(gray.std())
    (dark, bright), (warn_dark, warn_bright) = BRIGHTNESS_RANGE
    if mean < dark or mean > bright:
        issues.append(_issue("exposure", "error", "The photo is too dark or too bright. Retake it in even daylight, avoiding strong backlight."))
    elif mean < warn_dark or mean > warn_bright or contrast < MIN_CONTRAST:
        issues.append(_issue("exposure", "warning", "The lighting is uneven or flat. Even daylight gives the most realistic result."))

    return QualityReport(accepted=not any(i.severity == "error" for i in issues), issues=issues)
