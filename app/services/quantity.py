import math

from app.schemas.catalog import Material


def required_quantity(material: Material, measured: float) -> float:
    """Order quantity for a measured area (sq ft) or length (rft), including wastage."""
    spec = material.quantity
    with_waste = measured * (1 + spec.wastage_pct / 100)
    if spec.rule == "pieces":
        return float(math.ceil(with_waste / spec.unit_coverage_sqft))
    if spec.rule == "coverage":
        return math.ceil(round(with_waste / spec.unit_coverage_sqft * 10, 6)) / 10  # never order less than needed
    return round(with_waste, 1)
