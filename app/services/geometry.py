from dataclasses import dataclass
from functools import cached_property

import cv2
import numpy as np

from app.schemas.project import Plane, quad_area

CLICK_NOISE_PX = 2.0
UNCERTAINTY_SAMPLES = 100
BAND_PERCENTILES = (10, 90)
MAX_SAMPLE_POINTS = 5_000
MIN_COMPONENT_PX = 30
# Relative model error on top of click noise, measured against synthetic ground truth (docs/phase0-findings.md)
MODEL_ERROR = {False: 0.15, True: 0.03}


def _pixel_aspect(plane: Plane) -> float:
    """Width over height of the reference box as drawn on the photo."""
    q = np.asarray(plane.reference_quad, np.float32)
    edge = lambda a, b: float(np.linalg.norm(q[a] - q[b]))
    return (edge(1, 0) + edge(2, 3)) / (edge(3, 0) + edge(2, 1))


def real_width_ft(plane: Plane) -> float:
    """The width entered, or else the width the box implies for the height entered."""
    return plane.ref_width_ft or plane.ref_height_ft * _pixel_aspect(plane)


def reference_aspect_ratio(plane: Plane) -> float:
    """Photo shape of the reference over its real shape: ~1 when face-on, well below 1 when the surface is angled."""
    return _pixel_aspect(plane) / (real_width_ft(plane) / plane.ref_height_ft)


@dataclass(frozen=True)
class PlaneModel:
    """Maps image pixels on one surface to square feet using its reference rectangle."""

    quad: np.ndarray
    width_ft: float
    height_ft: float
    perspective: bool

    @classmethod
    def from_plane(cls, plane: Plane) -> "PlaneModel":
        return cls(np.asarray(plane.reference_quad, np.float32), real_width_ft(plane), plane.ref_height_ft, plane.perspective)

    @cached_property
    def scale(self) -> float:
        # ft per pixel from the reference's pixel area: robust to click noise on small references
        return float(np.sqrt(self.width_ft * self.height_ft / quad_area(self.quad.tolist())))

    @cached_property
    def _homography(self) -> np.ndarray:
        target = np.array([[0, 0], [self.width_ft, 0], [self.width_ft, self.height_ft], [0, self.height_ft]], np.float32)
        return cv2.getPerspectiveTransform(self.quad, target)

    def to_feet(self, points: np.ndarray) -> np.ndarray:
        if not self.perspective:
            return points * self.scale
        return cv2.perspectiveTransform(points.reshape(-1, 1, 2).astype(np.float32), self._homography).reshape(-1, 2)

    def area_ft2(self, points: np.ndarray | None, total: int) -> float:
        """Area of `total` mask pixels; `points` is a sample of them (needed only in perspective mode)."""
        if not total:
            return 0.0
        if not self.perspective:
            return total * self.scale**2
        h = self._homography
        w = h[2, 0] * points[:, 0] + h[2, 1] * points[:, 1] + h[2, 2]
        # a homography's local area scale is |det H| / w^3
        return float((abs(np.linalg.det(h)) / np.abs(w) ** 3).mean() * total)

    def perturbed(self, rng: np.random.Generator, count: int = UNCERTAINTY_SAMPLES) -> list["PlaneModel"]:
        """Copies with jittered reference corners, to estimate how click error propagates."""
        return [
            PlaneModel(self.quad + rng.normal(0, CLICK_NOISE_PX, self.quad.shape).astype(np.float32), self.width_ft, self.height_ft, self.perspective)
            for _ in range(count)
        ]


@dataclass(frozen=True)
class Measurement:
    area_ft2: float
    area_low: float
    area_high: float
    length_ft: float | None = None


def _sample_points(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    stride = max(1, len(xs) // MAX_SAMPLE_POINTS)
    return np.column_stack((xs[::stride], ys[::stride])).astype(np.float32)


def _longest_edge_ft(model: PlaneModel, contour: np.ndarray) -> float:
    box = model.to_feet(cv2.boxPoints(cv2.minAreaRect(contour)))
    return float(max(np.linalg.norm(box[i] - box[(i + 1) % 4]) for i in range(4)))


def _running_length_ft(model: PlaneModel, mask: np.ndarray) -> float:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return sum(_longest_edge_ft(model, c) for c in contours if cv2.contourArea(c) >= MIN_COMPONENT_PX)


def measure(mask: np.ndarray, model: PlaneModel, samples: list[PlaneModel], with_length: bool) -> Measurement:
    total = int(np.count_nonzero(mask))
    points = _sample_points(mask) if model.perspective and total else None  # not needed in scale mode
    estimate = model.area_ft2(points, total)
    low, high = np.percentile([m.area_ft2(points, total) for m in samples], BAND_PERCENTILES)
    margin = MODEL_ERROR[model.perspective]
    length = _running_length_ft(model, mask) if with_length else None
    return Measurement(estimate, min(float(low), estimate * (1 - margin)), max(float(high), estimate * (1 + margin)), length)
