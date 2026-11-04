from app.schemas.catalog import Category
from app.schemas.project import ProjectState, Region, quad_area
from app.services.catalog import get_catalog
from app.services.imaging import ImageError, mask_size
from app.services.scale import check_scale

MIN_PERSPECTIVE_REFERENCE = 0.02  # share of the photo; smaller references make the homography too noise-sensitive


class StateError(ValueError):
    pass


def validate_regions(regions: list[Region], image_size: tuple[int, int]) -> None:
    """Reject masks that are corrupt or don't match the photo's size."""
    for region in regions:
        try:
            size = mask_size(region.mask_png)
        except ImageError as exc:
            raise StateError(f"Region '{region.id}': {exc}") from exc
        if size != image_size:
            raise StateError(f"Region '{region.id}' mask is {size[0]}×{size[1]} but the photo is {image_size[0]}×{image_size[1]}.")


def validate_state(state: ProjectState, image_size: tuple[int, int]) -> None:
    """Reject saved states that reference missing planes/regions/materials or use unreliable calibration."""
    catalog = get_catalog()
    plane_ids = {p.id for p in state.planes}
    for plane in state.planes:
        if issue := check_scale(plane, image_size).issue:
            raise StateError(f"Surface '{plane.name}': {issue}")
        if plane.perspective and quad_area(plane.reference_quad) < MIN_PERSPECTIVE_REFERENCE * image_size[0] * image_size[1]:
            raise StateError(f"Plane '{plane.name}': perspective mode needs a reference covering at least {MIN_PERSPECTIVE_REFERENCE:.0%} of the photo.")
    regions = {r.id: r for r in state.regions}
    for region in state.regions:
        if region.plane_id not in plane_ids:
            raise StateError(f"Region '{region.id}' uses unknown plane '{region.plane_id}'.")

    for design in state.designs:
        for choices, is_glass in ((design.assignments, False), (design.glass, True)):
            for region_id, material_id in choices.items():
                region, material = regions.get(region_id), catalog.get(material_id)
                if region is None or material is None:
                    raise StateError(f"Design '{design.name}' references an unknown region or material.")
                if region.type not in material.applies_to or (material.category == Category.GLASS) != is_glass:
                    raise StateError(f"'{material.name}' cannot be applied to a {region.type.value} region here.")
    for material_id in state.rates:
        if material_id not in catalog:
            raise StateError(f"Unknown material '{material_id}' in rates.")
