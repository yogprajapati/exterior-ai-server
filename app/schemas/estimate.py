from pydantic import BaseModel

from app.schemas.catalog import Category
from app.schemas.project import RegionType


class RegionMeasure(BaseModel):
    region_id: str
    type: RegionType
    plane_id: str
    area_sqft: float
    area_low: float
    area_high: float
    length_ft: float | None = None


class TypeTotal(BaseModel):
    """Measurements rolled up per part type, for display and the report."""

    type: RegionType
    count: int
    area_sqft: float
    area_low: float
    area_high: float
    length_ft: float | None = None


class MaterialLine(BaseModel):
    material_id: str
    name: str
    category: Category
    region_ids: list[str]
    measured: float
    measured_unit: str
    quantity: float
    quantity_unit: str
    wastage_pct: float
    material_rate: float
    prep_rate: float
    labor_rate: float
    material_cost: float
    prep_cost: float
    labor_cost: float
    total_cost: float
    total_low: float
    total_high: float


class CategoryTotal(BaseModel):
    category: Category
    material_cost: float  # includes preparation materials
    labor_cost: float
    total_cost: float


class Estimate(BaseModel):
    design_id: str
    regions: list[RegionMeasure]
    type_totals: list[TypeTotal]
    lines: list[MaterialLine]
    categories: list[CategoryTotal]
    material_total: float  # includes preparation materials
    labor_total: float
    grand_total: float
    grand_total_low: float
    grand_total_high: float
