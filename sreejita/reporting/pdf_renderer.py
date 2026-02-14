# =====================================================
# EXECUTIVE PDF RENDERER — UNIVERSAL (PHASE 3 TRUST SAFE)
# Sreejita Framework v3.6
# =====================================================

from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import json

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.colors import HexColor
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    Table,
    TableStyle,
    PageBreak,
)
from reportlab.lib.units import inch
from reportlab.lib import utils


# =====================================================
# SAFE HELPERS
# =====================================================

def _safe_float(v, default=None):
    try:
        f = float(v)
        if 0.0 <= f <= 1.0:
            return f
    except Exception:
        pass
    return default


def _load_visual_metadata(path: Path) -> Dict[str, Any]:
    meta_path = path.with_suffix(".json")
    if meta_path.exists():
        try:
            return json.loads(meta_path.read_text())
        except Exception:
            pass
    return {}


def format_value(v: Any) -> str:
    """
    Phase-3 executive-safe numeric formatting.
    Avoids false precision.
    """
    if v is None:
        return "Insufficient data"

    try:
        v = float(v)
    except Exception:
        return "Insufficient data"

    if 0 <= v <= 1:
        return f"~{v * 100:.0f}%"

    if abs(v) >= 1_000_000:
        return f"~{v / 1_000_000:.1f}M"
    if abs(v) >= 1_000:
        return f"~{v / 1_000:.0f}K"

    return f"{v:.0f}"


# =====================================================
# EXECUTIVE PDF RENDERER
# =====================================================

class ExecutivePDFRenderer:
    """
    STRICTLY PRESENTATIONAL PDF RENDERER (PHASE 3)

    GUARANTEES:
    - Never computes intelligence
    - Never inflates certainty
    - Makes suppression visible
    - Board-safe under weak data
    """

    BORDER = HexColor("#e5e7eb")
    HEADER_BG = HexColor("#f3f4f6")

    # -------------------------------------------------
    # MAIN ENTRY
    # -------------------------------------------------
    def render(self, payload: Dict[str, Any], output_path: Path) -> Path:
        if not isinstance(payload, dict):
            raise RuntimeError("Invalid payload for PDF rendering")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=A4,
            leftMargin=40,
            rightMargin=40,
            topMargin=40,
            bottomMargin=40,
        )

        styles = getSampleStyleSheet()
        story: List[Any] = []

        # -------------------------------------------------
        # STYLES
        # -------------------------------------------------
        styles.add(ParagraphStyle(
            "SR_Title",
            fontSize=22,
            alignment=TA_CENTER,
            spaceAfter=18,
            fontName="Helvetica-Bold",
        ))

        styles.add(ParagraphStyle(
            "SR_Section",
            fontSize=15,
            spaceBefore=18,
            spaceAfter=10,
            fontName="Helvetica-Bold",
        ))

        styles.add(ParagraphStyle(
            "SR_Body",
            fontSize=11,
            leading=15,
            spaceAfter=6,
        ))

        styles.add(ParagraphStyle(
            "SR_Muted",
            fontSize=10,
            leading=14,
            textColor=HexColor("#6b7280"),
        ))

        # -------------------------------------------------
        # SAFE EXTRACTION
        # -------------------------------------------------
        executive = payload.get("executive") or {}
        visuals = payload.get("visuals") or []
        insights = payload.get("insights") or []
        recommendations = payload.get("recommendations") or []
        kpis = payload.get("kpis") or {}

        domain = str(payload.get("domain", "—")).replace("_", " ").title()

        # =================================================
        # PAGE 1 — EXECUTIVE OVERVIEW
        # =================================================
        story.append(Paragraph(
            "Sreejita Executive Intelligence Report",
            styles["SR_Title"],
        ))

        story.append(Paragraph(
            f"<b>Domain:</b> {domain}<br/>"
            f"<b>Generated:</b> {datetime.utcnow():%Y-%m-%d}",
            styles["SR_Body"],
        ))

        brief = executive.get("executive_brief")
        if isinstance(brief, str) and brief.strip():
            story.append(Spacer(1, 12))
            story.append(Paragraph("Executive Brief", styles["SR_Section"]))
            story.append(Paragraph(brief.strip(), styles["SR_Body"]))

        # =================================================
        # KPI TABLE
        # =================================================
        story.append(Spacer(1, 14))
        story.append(Paragraph("Key Performance Indicators", styles["SR_Section"]))

        if not kpis:
            story.append(Paragraph(
                "Insufficient data to compute reliable KPIs.",
                styles["SR_Muted"],
            ))
        else:
            rows = [["Metric", "Value"]]
            for k, v in list(kpis.items())[:8]:
                rows.append([k.replace("_", " ").title(), format_value(v)])

            table = Table(rows, colWidths=[4.5 * inch, 2.5 * inch])
            table.setStyle(TableStyle([
                ("GRID", (0, 0), (-1, -1), 0.5, self.BORDER),
                ("BACKGROUND", (0, 0), (-1, 0), self.HEADER_BG),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("PADDING", (0, 0), (-1, -1), 8),
            ]))
            story.append(table)

        # =================================================
        # VISUAL EVIDENCE
        # =================================================
        valid_visuals = [
            v for v in visuals
            if isinstance(v, dict) and Path(v.get("path", "")).exists()
        ][:6]

        if valid_visuals:
            story.append(PageBreak())
            story.append(Paragraph("Visual Evidence", styles["SR_Section"]))

            for v in valid_visuals:
                path = Path(v["path"])
                meta = _load_visual_metadata(path)

                try:
                    img = utils.ImageReader(str(path))
                    iw, ih = img.getSize()
                    w = 6 * inch
                    h = min(w * ih / iw, 4 * inch)

                    story.append(Image(str(path), width=w, height=h))

                    if meta.get("status") != "rendered":
                        story.append(Paragraph(
                            f"<b>Insufficient data:</b> {meta.get('reason','')}",
                            styles["SR_Muted"],
                        ))
                    else:
                        story.append(Paragraph(
                            v.get("caption", "Visual evidence"),
                            styles["SR_Body"],
                        ))

                    story.append(Spacer(1, 12))
                except Exception:
                    continue

        # =================================================
        # INSIGHTS
        # =================================================
        story.append(PageBreak())
        story.append(Paragraph("Key Insights", styles["SR_Section"]))

        if not insights:
            story.append(Paragraph(
                "No reliable insights generated due to weak or insufficient signal.",
                styles["SR_Muted"],
            ))
        else:
            for ins in insights[:5]:
                story.append(Paragraph(
                    f"<b>{ins.get('title','')}</b>",
                    styles["SR_Body"],
                ))
                story.append(Paragraph(
                    ins.get("so_what", ""),
                    styles["SR_Body"],
                ))
                story.append(Spacer(1, 8))

        # =================================================
        # RECOMMENDATIONS
        # =================================================
        story.append(PageBreak())
        story.append(Paragraph("Recommendations", styles["SR_Section"]))

        if not recommendations:
            story.append(Paragraph(
                "No recommendations issued due to insufficient supporting evidence.",
                styles["SR_Muted"],
            ))
        else:
            for rec in recommendations[:5]:
                story.append(Paragraph(
                    f"<b>{rec.get('action','')}</b>",
                    styles["SR_Body"],
                ))
                story.append(Paragraph(
                    f"Owner: {rec.get('owner','—')} | "
                    f"Timeline: {rec.get('timeline','—')}",
                    styles["SR_Muted"],
                ))
                story.append(Spacer(1, 10))

        doc.build(story)
        return output_path
