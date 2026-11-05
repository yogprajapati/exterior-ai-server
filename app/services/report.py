import io
from datetime import date
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from app.schemas.estimate import Estimate
from app.schemas.project import Design
from app.services.catalog import DATA_DIR, get_catalog

FONT, FONT_BOLD = "DejaVu", "DejaVu-Bold"
pdfmetrics.registerFont(TTFont(FONT, str(DATA_DIR / "fonts" / "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont(FONT_BOLD, str(DATA_DIR / "fonts" / "DejaVuSans-Bold.ttf")))

PAGE_WIDTH = A4[0] - 30 * mm
ACCENT = colors.HexColor("#1f4e5f")
METHOD_NOTES = (
    "Each surface is calibrated from a reference rectangle of known real size marked on the photo; pixel areas are converted to square feet from it.",
    "Overlapping regions are resolved so openings (windows, doors) and railings are excluded from wall areas.",
    "Ranges reflect the uncertainty from placing the reference corners (about ±2 px). Photo perspective and surfaces at a different depth add error.",
    "Order quantity = measured area × (1 + wastage) ÷ coverage per unit (paint per litre, tiles per piece); other materials are bought by the sq ft or running foot.",
    "Cost = order quantity × material rate + measured area × preparation rate (primer, putty, adhesive, fixings) + measured area × labour rate. Wastage is paid for in material only.",
    "Rates are editable planning assumptions in INR, not quotations. This estimate is advisory and not legally binding; confirm with your contractor.",
)


def format_inr(value: float) -> str:
    """Indian digit grouping, e.g. ₹1,23,456."""
    digits = f"{round(value):d}"
    head, tail = digits[:-3], digits[-3:]
    groups = []
    while len(head) > 2:
        groups.insert(0, head[-2:])
        head = head[:-2]
    return "₹" + ",".join(([head] if head else []) + groups + [tail])


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()["Normal"]
    return {
        "title": ParagraphStyle("title", base, fontName=FONT_BOLD, fontSize=18, leading=24, textColor=ACCENT, spaceAfter=2),
        "h2": ParagraphStyle("h2", base, fontName=FONT_BOLD, fontSize=12, textColor=ACCENT, spaceBefore=10, spaceAfter=4),
        "body": ParagraphStyle("body", base, fontName=FONT, fontSize=8.5, leading=11),
        "cell": ParagraphStyle("cell", base, fontName=FONT, fontSize=8, leading=10),
    }


def _table(rows: list[list], widths: list[float], bold_last: bool = False) -> Table:
    table = Table(rows, colWidths=widths, repeatRows=1)
    style = [
        ("FONTNAME", (0, 0), (-1, -1), FONT), ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("BACKGROUND", (0, 0), (-1, 0), ACCENT), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("FONTNAME", (0, 0), (-1, 0), FONT_BOLD),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f6f7")]),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#c9d3d6")), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]
    if bold_last:
        style.append(("FONTNAME", (0, -1), (-1, -1), FONT_BOLD))
    table.setStyle(TableStyle(style))
    return table


def _image(data: bytes, width: float) -> Image:
    image = Image(io.BytesIO(data))
    image.drawHeight = width * image.imageHeight / image.imageWidth
    image.drawWidth = width
    return image


def build_report(project_id: str, design: Design, original_jpeg: bytes, render_jpeg: bytes, estimate: Estimate) -> bytes:
    styles, catalog = _styles(), get_catalog()
    region_type = {r.region_id: r.type.value.replace("_", " ") for r in estimate.regions}
    story = [
        Paragraph("Exterior Renovation Estimate", styles["title"]),
        Paragraph(f"Design: {escape(design.name)} &nbsp;·&nbsp; Project {project_id[:8]} &nbsp;·&nbsp; {date.today():%d %b %Y}", styles["body"]),
        Spacer(1, 6),
    ]
    half = PAGE_WIDTH / 2 - 3
    story.append(Table(
        [[_image(original_jpeg, half), _image(render_jpeg, half)], [Paragraph("Original", styles["body"]), Paragraph("Redesigned", styles["body"])]],
        colWidths=[half + 3, half + 3],
    ))

    story.append(Paragraph("Selected materials", styles["h2"]))
    rows = [["Material", "Applied to", "Durability & upkeep"]]
    for line in estimate.lines:
        applied = ", ".join(sorted({region_type[r] for r in line.region_ids}))
        rows.append([Paragraph(line.name, styles["cell"]), Paragraph(applied, styles["cell"]), Paragraph(catalog[line.material_id].info, styles["cell"])])
    story.append(_table(rows, [PAGE_WIDTH * 0.28, PAGE_WIDTH * 0.2, PAGE_WIDTH * 0.52]))

    story.append(Paragraph("Quantities and cost", styles["h2"]))
    rows = [["Material", "Measured", "Order quantity", "Material", "Preparation", "Labour", "Total"]]
    for l in estimate.lines:
        quantity = f"{l.quantity:,.1f} {l.quantity_unit}" + (f" (+{l.wastage_pct:g}% wastage)" if l.wastage_pct else "")
        rows.append([Paragraph(l.name, styles["cell"]), f"{l.measured:,.1f} {l.measured_unit}", Paragraph(quantity, styles["cell"]), format_inr(l.material_cost),
                     format_inr(l.prep_cost), format_inr(l.labor_cost), format_inr(l.total_cost)])
    story.append(_table(rows, [PAGE_WIDTH * w for w in (0.24, 0.12, 0.17, 0.12, 0.12, 0.12, 0.11)]))

    story.append(Paragraph("Cost by category", styles["h2"]))
    rows = [["Category", "Material + preparation", "Labour", "Total"]]
    rows += [[c.category.value.replace("_", " ").title(), format_inr(c.material_cost), format_inr(c.labor_cost), format_inr(c.total_cost)] for c in estimate.categories]
    rows.append(["Grand total", format_inr(estimate.material_total), format_inr(estimate.labor_total), format_inr(estimate.grand_total)])
    story.append(_table(rows, [PAGE_WIDTH * w for w in (0.34, 0.26, 0.2, 0.2)], bold_last=True))
    story.append(Spacer(1, 4))
    story.append(Paragraph(f"Likely range given measurement uncertainty: {format_inr(estimate.grand_total_low)} – {format_inr(estimate.grand_total_high)}", styles["body"]))

    story.append(Paragraph("How these numbers were derived", styles["h2"]))
    story += [Paragraph(f"• {note}", styles["body"]) for note in METHOD_NOTES]

    buffer = io.BytesIO()
    SimpleDocTemplate(buffer, pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm, topMargin=15 * mm, bottomMargin=15 * mm,
                      title="Exterior Renovation Estimate").build(story)
    return buffer.getvalue()
