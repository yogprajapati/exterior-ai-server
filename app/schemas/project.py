from collections.abc import Sequence
from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

Point = tuple[float, float]
MIN_REFERENCE_AREA_PX = 100.0


class RegionType(StrEnum):
    WALL = "wall"
    WINDOW = "window"
    DOOR = "door"
    BALCONY = "balcony"
    RAILING = "railing"
    PILLAR = "pillar"
    PARAPET = "parapet"
    GATE = "gate"
    ROOF_EDGE = "roof_edge"


# Earlier entries win where masks overlap, so openings and railings cut out of walls
REGION_PRIORITY: tuple[RegionType, ...] = (
    RegionType.DOOR,
    RegionType.WINDOW,
    RegionType.RAILING,
    RegionType.GATE,
    RegionType.PILLAR,
    RegionType.BALCONY,
    RegionType.PARAPET,
    RegionType.ROOF_EDGE,
    RegionType.WALL,
)


def quad_area(quad: Sequence[Sequence[float]]) -> float:
    return abs(sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(quad, [*quad[1:], quad[0]]))) / 2


def is_upright_clockwise(quad: list[Point]) -> bool:
    """Convex, ordered TL, TR, BR, BL on screen (y down), with the top edge above the bottom edge."""
    edges = [(b[0] - a[0], b[1] - a[1]) for a, b in zip(quad, quad[1:] + quad[:1])]
    crosses = [e[0] * n[1] - e[1] * n[0] for e, n in zip(edges, edges[1:] + edges[:1])]
    (tlx, tly), (trx, try_), (brx, bry), (blx, bly) = quad
    return all(c > 0 for c in crosses) and tly + try_ < bly + bry and tlx + blx < trx + brx


class Plane(BaseModel):
    """A surface calibrated by a reference rectangle (corners TL, TR, BR, BL) of known real size."""

    id: str
    name: str
    reference_quad: list[Point] = Field(min_length=4, max_length=4)
    ref_height_ft: float = Field(gt=0)
    ref_width_ft: float | None = Field(default=None, gt=0)  # when absent the box's own proportions give it (face-on surfaces)
    perspective: bool = False

    @model_validator(mode="after")
    def _check_quad(self) -> "Plane":
        if self.perspective and self.ref_width_ft is None:
            raise ValueError("a surface photographed at an angle needs the real width as well as the height")
        if quad_area(self.reference_quad) < MIN_REFERENCE_AREA_PX or not is_upright_clockwise(self.reference_quad):
            raise ValueError("reference_quad must be an upright quadrilateral with corners ordered TL, TR, BR, BL")
        return self


class ScaleCheck(BaseModel):
    """What a reference box implies for the whole photo, shown to the user before they continue."""

    ft_per_pixel: float
    photo_width_ft: float
    photo_height_ft: float
    issue: str | None = None


class Region(BaseModel):
    model_config = ConfigDict(ser_json_bytes="base64", val_json_bytes="base64")

    id: str
    type: RegionType
    plane_id: str
    mask_png: bytes


class Design(BaseModel):
    id: str
    name: str
    assignments: dict[str, str] = {}  # region id -> material id (a window's frame)
    glass: dict[str, str] = {}  # window region id -> glass material id


class Rates(BaseModel):
    material_rate: float = Field(ge=0)  # per ordered unit (litre, piece, sq ft, rft)
    labor_rate: float = Field(ge=0)  # per measured sq ft / rft
    prep_rate: float = Field(default=0, ge=0)  # primer, putty, adhesive, fixings: per measured sq ft / rft


class ProjectState(BaseModel):
    planes: list[Plane] = []
    regions: list[Region] = []
    designs: list[Design] = []
    rates: dict[str, Rates] = {}  # material id -> user override


class ProjectUpdate(BaseModel):
    planes: list[Plane] | None = None
    regions: list[Region] | None = None
    designs: list[Design] | None = None
    rates: dict[str, Rates] | None = None


class QualityIssue(BaseModel):
    code: str
    severity: str  # "error" | "warning"
    message: str


class QualityReport(BaseModel):
    accepted: bool
    issues: list[QualityIssue] = []


class ProjectCreated(BaseModel):
    id: str
    width: int
    height: int
    quality: QualityReport


class ProjectOut(ProjectState):
    id: str
    width: int
    height: int
    created_at: datetime
    updated_at: datetime
