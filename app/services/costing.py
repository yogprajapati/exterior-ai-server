from app.schemas.catalog import Material
from app.schemas.project import Rates


def effective_rates(material: Material, overrides: dict[str, Rates]) -> Rates:
    return overrides.get(material.id, material.rates)


def line_costs(rates: Rates, quantity: float, measured: float) -> tuple[float, float, float]:
    """(material, preparation, labour): material per ordered unit incl. wastage; preparation and labour per measured sq ft / rft."""
    return round(quantity * rates.material_rate, 2), round(measured * rates.prep_rate, 2), round(measured * rates.labor_rate, 2)
