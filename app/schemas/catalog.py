from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, model_validator

from app.schemas.project import Rates, RegionType


class Category(StrEnum):
    PAINT = "paint"
    STONE_CLADDING = "stone_cladding"
    TILES = "tiles"
    TEXTURE_FINISH = "texture_finish"
    PANELS = "panels"
    WINDOW_FRAMES = "window_frames"
    GLASS = "glass"
    DOORS = "doors"
    GATES = "gates"
    RAILINGS = "railings"


class RenderSpec(BaseModel):
    """How a material looks. `color` is also what the redesign is calibrated to. paint/texture cover a surface; tint recolours it and keeps what shows through; glass tints and reflects."""

    mode: Literal["paint", "texture", "tint", "glass"]
    color: str | None = None  # paint colour, tint colour, or glass tint
    texture: str | None = None
    tile_ft: float = 6.5  # real-world size of one texture repeat
    relief: float = 0.5  # how much of the photo's fine detail shows through
    strength: float = 1.0  # how far a tint moves the original colours (0 = unchanged)
    gloss: float = 0.0  # sky reflection on glass (0 = none)

    @model_validator(mode="after")
    def _check_inputs(self) -> "RenderSpec":
        if self.mode == "texture" and not self.texture:
            raise ValueError("texture mode needs a texture file")
        if self.mode != "texture" and not self.color:
            raise ValueError(f"{self.mode} mode needs a colour")
        return self


class QuantitySpec(BaseModel):
    measure: Literal["area", "length"]
    rule: Literal["coverage", "pieces", "direct"]
    unit: str
    unit_coverage_sqft: float | None = None  # sq ft per litre ("coverage") or per piece ("pieces")
    wastage_pct: float
    share: float = 1.0  # part of the region's measure this material covers (glass is most of a window's opening)

    @model_validator(mode="after")
    def _check_coverage(self) -> "QuantitySpec":
        if self.rule != "direct" and not self.unit_coverage_sqft:
            raise ValueError(f"rule '{self.rule}' needs unit_coverage_sqft")
        return self


class Material(BaseModel):
    id: str
    name: str
    category: Category
    applies_to: list[RegionType]
    look: str  # how the image editor is told to describe it
    render: RenderSpec
    quantity: QuantitySpec
    rates: Rates
    info: str
