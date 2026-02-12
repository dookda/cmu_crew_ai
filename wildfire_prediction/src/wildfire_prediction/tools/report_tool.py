"""Research report formatting and saving tool."""

import os
from typing import Any

from crewai.tools import tool


REQUIRED_SECTIONS = [
    "Abstract",
    "Introduction",
    "Study Area",
    "Data and Methods",
    "Results",
    "Discussion",
    "Conclusion",
    "References",
]


@tool("report_formatting_tool")
def report_formatting_tool(report_content: str) -> str:
    """Validate, format, and save the research report as Markdown.

    Args:
        report_content: Full research report in Markdown format.

    Returns:
        Confirmation with word count and validation results.
    """
    try:
        os.makedirs("output", exist_ok=True)

        # Validate required sections
        missing_sections = []
        for section in REQUIRED_SECTIONS:
            if section.lower() not in report_content.lower():
                missing_sections.append(section)

        # Word count
        words = report_content.split()
        word_count = len(words)

        # Save report
        output_path = "output/research_report.md"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report_content)

        # Build confirmation
        report_lines = [
            "=" * 60,
            "REPORT SAVE CONFIRMATION",
            "=" * 60,
            f"Report saved to: {output_path}",
            f"Word count: {word_count}",
            f"Character count: {len(report_content)}",
            "",
        ]

        if missing_sections:
            report_lines.append(f"WARNING: Missing sections: {', '.join(missing_sections)}")
        else:
            report_lines.append("All required sections present: OK")

        report_lines.append("")
        report_lines.append("Sections found:")
        for section in REQUIRED_SECTIONS:
            status = "FOUND" if section.lower() in report_content.lower() else "MISSING"
            report_lines.append(f"  [{status}] {section}")

        report_lines.append("")
        report_lines.append("=" * 60)

        return "\n".join(report_lines)

    except Exception as e:
        return f"Error saving report: {str(e)}"
