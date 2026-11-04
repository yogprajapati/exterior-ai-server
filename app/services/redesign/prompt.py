from collections import defaultdict

from app.schemas.catalog import Material
from app.schemas.project import RegionType
from app.services.layering import RegionMask

SUBJECT = {
    RegionType.WALL: "the exterior wall surfaces",
    RegionType.PILLAR: "the pillars and columns",
    RegionType.PARAPET: "the parapet walls",
    RegionType.ROOF_EDGE: "the cornice and roof-edge band",
    RegionType.BALCONY: "the balcony slabs and fronts",
    RegionType.WINDOW: "the window frames",
    RegionType.DOOR: "the doors",
    RegionType.GATE: "the gates",
    RegionType.RAILING: "the railings",
}
KEEP = (
    "Keep the building's shape and proportions, every window, door, balcony, railing, gate, the roof, plants, sky and surroundings exactly the same. "
    "Do not add, remove or move anything. Realistic photograph, natural light."
)


def _dominant(regions: list[RegionMask], materials: dict[str, Material]) -> Material:
    """The material covering most of the area, when one part type is given several."""
    area: dict[str, int] = defaultdict(int)
    chosen: dict[str, Material] = {}
    for region in regions:
        material = materials[region.id]
        area[material.id] += int(region.mask.sum())
        chosen[material.id] = material
    return chosen[max(area, key=area.get)]


def build_prompt(regions: list[RegionMask], frames: dict[str, Material], glass: dict[str, Material]) -> str:
    """The edit instruction: one clause per part type the design changes (`frames` maps region id to a frame or finish, `glass` to a window glass)."""
    by_type: dict[RegionType, list[RegionMask]] = defaultdict(list)
    for region in regions:
        if region.id in frames:
            by_type[region.type].append(region)
    clauses = []
    for region_type, group in by_type.items():
        look = _dominant(group, frames).look
        clauses.append(f"Replace {SUBJECT[region_type]} with {look}")
    if windows := [r for r in regions if r.type == RegionType.WINDOW and r.id in glass]:
        clauses.append(f"Use {_dominant(windows, glass).look} in the windows")
    return ". ".join(clauses) + ". " + KEEP
