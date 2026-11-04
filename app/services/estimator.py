from dataclasses import dataclass, field

import numpy as np

from app.schemas.catalog import Category, Material
from app.schemas.estimate import (
    CategoryTotal,
    Estimate,
    MaterialLine,
    RegionMeasure,
    TypeTotal,
)
from app.schemas.project import Design, ProjectState, RegionType
from app.services import costing, quantity
from app.services.catalog import get_catalog
from app.services.geometry import Measurement, PlaneModel, measure
from app.services.layering import RegionMask

SEED = 0
LENGTH_MEASURED = (RegionType.RAILING, RegionType.BALCONY)  # railings are priced per running foot, along a balcony's front edge


@dataclass
class _Group:
    material: Material
    region_ids: list[str] = field(default_factory=list)
    measured: float = 0.0
    low: float = 0.0
    high: float = 0.0


def measure_regions(state: ProjectState, masks: list[RegionMask]) -> list[tuple[str, RegionType, str, Measurement]]:
    """Measure every region on its own plane, with overlaps already resolved."""
    rng = np.random.default_rng(SEED)
    models = {p.id: PlaneModel.from_plane(p) for p in state.planes}
    samples = {plane_id: model.perturbed(rng) for plane_id, model in models.items()}
    return [
        (r.id, r.type, r.plane_id, measure(r.mask, models[r.plane_id], samples[r.plane_id], r.type in LENGTH_MEASURED))
        for r in masks
    ]


def _type_totals(measured: list[tuple[str, RegionType, str, Measurement]]) -> list[TypeTotal]:
    totals: dict[RegionType, TypeTotal] = {}
    for _, region_type, _, m in measured:
        total = totals.setdefault(region_type, TypeTotal(type=region_type, count=0, area_sqft=0, area_low=0, area_high=0))
        total.count += 1
        total.area_sqft += m.area_ft2
        total.area_low += m.area_low
        total.area_high += m.area_high
        if m.length_ft is not None:
            total.length_ft = (total.length_ft or 0) + m.length_ft
    for total in totals.values():
        total.area_sqft, total.area_low, total.area_high = (round(v, 1) for v in (total.area_sqft, total.area_low, total.area_high))
        total.length_ft = None if total.length_ft is None else round(total.length_ft, 1)
    return list(totals.values())


def _measured_values(material: Material, m: Measurement) -> tuple[float, float, float]:
    if material.quantity.measure == "area":
        share = material.quantity.share
        return m.area_ft2 * share, m.area_low * share, m.area_high * share
    length = m.length_ft or 0.0
    ratio = (np.sqrt(m.area_low / m.area_ft2), np.sqrt(m.area_high / m.area_ft2)) if m.area_ft2 else (1.0, 1.0)
    return length, length * ratio[0], length * ratio[1]


def _line(group: _Group, state: ProjectState) -> MaterialLine:
    material = group.material
    rates = costing.effective_rates(material, state.rates)
    qty = quantity.required_quantity(material, group.measured)
    material_cost, prep_cost, labor_cost = costing.line_costs(rates, qty, group.measured)
    total = material_cost + prep_cost + labor_cost
    low_share = group.low / group.measured if group.measured else 1.0
    high_share = group.high / group.measured if group.measured else 1.0
    return MaterialLine(
        material_id=material.id,
        name=material.name,
        category=material.category,
        region_ids=group.region_ids,
        measured=round(group.measured, 1),
        measured_unit="rft" if material.quantity.measure == "length" else "sq ft",
        quantity=qty,
        quantity_unit=material.quantity.unit,
        wastage_pct=material.quantity.wastage_pct,
        material_rate=rates.material_rate,
        prep_rate=rates.prep_rate,
        labor_rate=rates.labor_rate,
        material_cost=material_cost,
        prep_cost=prep_cost,
        labor_cost=labor_cost,
        total_cost=round(total, 2),
        total_low=round(total * low_share, 2),
        total_high=round(total * high_share, 2),
    )


def build_estimate(state: ProjectState, design: Design, masks: list[RegionMask]) -> Estimate:
    catalog = get_catalog()
    measured = measure_regions(state, masks)
    by_region = {region_id: m for region_id, _, _, m in measured}
    groups: dict[str, _Group] = {}
    for region_id, material_id in (*design.assignments.items(), *design.glass.items()):
        material = catalog[material_id]
        value, low, high = _measured_values(material, by_region[region_id])
        group = groups.setdefault(material_id, _Group(material))
        group.region_ids.append(region_id)
        group.measured += value
        group.low += low
        group.high += high

    lines = [_line(g, state) for g in groups.values()]
    totals: dict[Category, CategoryTotal] = {}
    for line in lines:
        t = totals.setdefault(line.category, CategoryTotal(category=line.category, material_cost=0, labor_cost=0, total_cost=0))
        t.material_cost = round(t.material_cost + line.material_cost + line.prep_cost, 2)
        t.labor_cost = round(t.labor_cost + line.labor_cost, 2)
        t.total_cost = round(t.total_cost + line.total_cost, 2)
    return Estimate(
        design_id=design.id,
        regions=[
            RegionMeasure(region_id=i, type=t, plane_id=p, area_sqft=round(m.area_ft2, 1), area_low=round(m.area_low, 1),
                          area_high=round(m.area_high, 1), length_ft=None if m.length_ft is None else round(m.length_ft, 1))
            for i, t, p, m in measured
        ],
        type_totals=_type_totals(measured),
        lines=lines,
        categories=list(totals.values()),
        material_total=round(sum(l.material_cost + l.prep_cost for l in lines), 2),
        labor_total=round(sum(l.labor_cost for l in lines), 2),
        grand_total=round(sum(l.total_cost for l in lines), 2),
        grand_total_low=round(sum(l.total_low for l in lines), 2),
        grand_total_high=round(sum(l.total_high for l in lines), 2),
    )
