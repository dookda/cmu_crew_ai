"""เครื่องมือจัดรูปแบบและบันทึกรายงานวิจัย"""

import os
from typing import Any

from crewai.tools import tool


# หัวข้อที่จำเป็นต้องมีในรายงาน
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
    """ตรวจสอบ จัดรูปแบบ และบันทึกรายงานวิจัยเป็น Markdown

    Args:
        report_content: เนื้อหารายงานวิจัยฉบับเต็มในรูปแบบ Markdown

    Returns:
        ข้อความยืนยันพร้อมจำนวนคำและผลการตรวจสอบ
    """
    try:
        os.makedirs("output", exist_ok=True)

        # ตรวจสอบหัวข้อที่จำเป็น
        missing_sections = []
        for section in REQUIRED_SECTIONS:
            if section.lower() not in report_content.lower():
                missing_sections.append(section)

        # นับจำนวนคำ
        words = report_content.split()
        word_count = len(words)

        # บันทึกรายงาน
        output_path = "output/research_report.md"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report_content)

        # สร้างข้อความยืนยัน
        report_lines = [
            "=" * 60,
            "ยืนยันการบันทึกรายงาน (REPORT SAVE CONFIRMATION)",
            "=" * 60,
            f"บันทึกรายงานที่: {output_path}",
            f"จำนวนคำ: {word_count}",
            f"จำนวนตัวอักษร: {len(report_content)}",
            "",
        ]

        if missing_sections:
            report_lines.append(f"คำเตือน: หัวข้อที่ขาดหาย: {', '.join(missing_sections)}")
        else:
            report_lines.append("หัวข้อครบถ้วนทั้งหมด: OK")

        report_lines.append("")
        report_lines.append("ผลการตรวจสอบหัวข้อ:")
        for section in REQUIRED_SECTIONS:
            status = "พบ" if section.lower() in report_content.lower() else "ไม่พบ"
            report_lines.append(f"  [{status}] {section}")

        report_lines.append("")
        report_lines.append("=" * 60)

        return "\n".join(report_lines)

    except Exception as e:
        return f"เกิดข้อผิดพลาดระหว่างบันทึกรายงาน: {str(e)}"
