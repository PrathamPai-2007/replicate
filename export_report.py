from __future__ import annotations

import argparse
import html
import re
from pathlib import Path

from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer


NUMBERED_ITEM = re.compile(r"^(\d+)\.\s+(.*)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the replication/adaptation report to PDF and RTF."
    )
    parser.add_argument(
        "--input",
        default="REPLICATION_ADAPTATION_REPORT.md",
        help="Markdown report path.",
    )
    parser.add_argument(
        "--output-stem",
        default="reports/Replication_Adaptation_Report",
        help="Output path without extension.",
    )
    return parser.parse_args()


def build_styles():
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Title"],
            alignment=TA_CENTER,
            fontSize=20,
            leading=24,
            spaceAfter=18,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Body",
            parent=styles["BodyText"],
            fontSize=10.5,
            leading=14,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BulletBody",
            parent=styles["BodyText"],
            fontSize=10.5,
            leading=14,
            leftIndent=10,
            spaceAfter=2,
        )
    )
    styles["Heading1"].spaceBefore = 12
    styles["Heading1"].spaceAfter = 8
    styles["Heading2"].spaceBefore = 10
    styles["Heading2"].spaceAfter = 6
    return styles


def _escape_inline(text: str) -> str:
    return html.escape(text).replace("`", "")


def build_pdf(markdown_text: str, output_path: Path) -> None:
    styles = build_styles()
    story = []
    bullet_items = []
    numbered_items = []

    def flush_lists() -> None:
        nonlocal bullet_items, numbered_items
        if bullet_items:
            story.append(
                ListFlowable(
                    bullet_items,
                    bulletType="bullet",
                    leftIndent=18,
                    bulletFontName="Helvetica",
                )
            )
            story.append(Spacer(1, 0.08 * inch))
            bullet_items = []
        if numbered_items:
            story.append(
                ListFlowable(
                    numbered_items,
                    bulletType="1",
                    leftIndent=18,
                    bulletFontName="Helvetica",
                )
            )
            story.append(Spacer(1, 0.08 * inch))
            numbered_items = []

    for raw_line in markdown_text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()

        if not stripped:
            flush_lists()
            story.append(Spacer(1, 0.08 * inch))
            continue

        if stripped.startswith("# "):
            flush_lists()
            story.append(Paragraph(_escape_inline(stripped[2:]), styles["ReportTitle"]))
            continue

        if stripped.startswith("## "):
            flush_lists()
            story.append(Paragraph(_escape_inline(stripped[3:]), styles["Heading1"]))
            continue

        if stripped.startswith("### "):
            flush_lists()
            story.append(Paragraph(_escape_inline(stripped[4:]), styles["Heading2"]))
            continue

        if stripped.startswith("- "):
            bullet_items.append(ListItem(Paragraph(_escape_inline(stripped[2:]), styles["BulletBody"])))
            continue

        numbered_match = NUMBERED_ITEM.match(stripped)
        if numbered_match:
            numbered_items.append(ListItem(Paragraph(_escape_inline(numbered_match.group(2)), styles["BulletBody"])))
            continue

        if stripped.startswith("```"):
            flush_lists()
            continue

        flush_lists()
        story.append(Paragraph(_escape_inline(stripped), styles["Body"]))

    flush_lists()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=0.7 * inch,
        leftMargin=0.7 * inch,
        topMargin=0.7 * inch,
        bottomMargin=0.7 * inch,
    )
    document.build(story)


def _rtf_escape(text: str) -> str:
    return (
        text.replace("\\", "\\\\")
        .replace("{", "\\{")
        .replace("}", "\\}")
        .replace("\t", "\\tab ")
    )


def build_rtf(markdown_text: str, output_path: Path) -> None:
    lines = [
        r"{\rtf1\ansi\deff0",
        r"{\fonttbl{\f0 Calibri;}{\f1 Arial;}}",
        r"\paperw11907\paperh16840\margl1134\margr1134\margt1134\margb1134",
        r"\fs22",
    ]

    for raw_line in markdown_text.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            lines.append(r"\par")
            continue

        if stripped.startswith("# "):
            lines.append(r"\pard\qc\b\f1\fs32 " + _rtf_escape(stripped[2:]) + r"\b0\fs22\par")
            continue
        if stripped.startswith("## "):
            lines.append(r"\pard\b\f1\fs26 " + _rtf_escape(stripped[3:]) + r"\b0\fs22\par")
            continue
        if stripped.startswith("### "):
            lines.append(r"\pard\b\f1\fs24 " + _rtf_escape(stripped[4:]) + r"\b0\fs22\par")
            continue
        if stripped.startswith("- "):
            lines.append(r"\pard\li360\u8226? " + _rtf_escape(stripped[2:]) + r"\par")
            continue

        numbered_match = NUMBERED_ITEM.match(stripped)
        if numbered_match:
            lines.append(r"\pard\li360 " + _rtf_escape(numbered_match.group(1) + ". " + numbered_match.group(2)) + r"\par")
            continue

        if stripped.startswith("```"):
            continue

        lines.append(r"\pard " + _rtf_escape(stripped.replace("`", "")) + r"\par")

    lines.append("}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input report not found: {input_path}")

    text = input_path.read_text(encoding="utf-8")
    output_stem = Path(args.output_stem)
    pdf_path = output_stem.with_suffix(".pdf")
    rtf_path = output_stem.with_suffix(".rtf")

    build_pdf(text, pdf_path)
    build_rtf(text, rtf_path)

    print(f"wrote_pdf={pdf_path}")
    print(f"wrote_rtf={rtf_path}")


if __name__ == "__main__":
    main()
