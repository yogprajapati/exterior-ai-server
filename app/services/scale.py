from app.schemas.project import Plane, ScaleCheck
from app.services.geometry import PlaneModel, reference_aspect_ratio

FACE_ON_ASPECT_RANGE = (0.75, 1.33)  # scale-only calibration needs a reference that keeps its real shape
PHOTO_WIDTH_FT_RANGE = (12, 300)  # a photo of a house front shows roughly this much, left to right


def scale_issue(plane: Plane, photo_width_ft: float) -> str | None:
    """Why this reference box cannot be right, in plain words; None when it looks plausible."""
    shown = f"{plane.ref_height_ft:g} ft tall" if plane.ref_width_ft is None else f"{plane.ref_width_ft:g} × {plane.ref_height_ft:g} ft"
    if plane.ref_width_ft is not None and not plane.perspective and not FACE_ON_ASPECT_RANGE[0] <= reference_aspect_ratio(plane) <= FACE_ON_ASPECT_RANGE[1]:
        return f"The box is not shaped like a {shown} rectangle. Make sure it surrounds only what you chose, or change the size. If that wall is photographed at an angle, open \"Is this wall at a sharp angle\" and tick the box, then box something larger."
    if photo_width_ft < PHOTO_WIDTH_FT_RANGE[0]:
        return f"With this box and size, the whole photo would show only about {photo_width_ft:.0f} ft across, too small for a house. Is the box around more than the one thing that is {shown}?"
    if photo_width_ft > PHOTO_WIDTH_FT_RANGE[1]:
        return f"With this box and size, the whole photo would show about {photo_width_ft:.0f} ft across, wider than a house. Is the size you entered too large for the box?"
    return None


def check_scale(plane: Plane, image_size: tuple[int, int]) -> ScaleCheck:
    scale = PlaneModel.from_plane(plane).scale
    width_ft, height_ft = image_size[0] * scale, image_size[1] * scale
    return ScaleCheck(ft_per_pixel=scale, photo_width_ft=width_ft, photo_height_ft=height_ft, issue=scale_issue(plane, width_ft))
