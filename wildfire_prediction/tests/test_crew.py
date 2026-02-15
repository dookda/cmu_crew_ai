"""ทดสอบพื้นฐานสำหรับ Wildfire Prediction Crew"""

import os

import pytest


def test_sample_data_exists():
    """ตรวจว่าไฟล์ข้อมูลตัวอย่างมีอยู่จริง"""
    assert os.path.exists("data/sample_hotspot_data.csv")


def test_crew_import():
    """ตรวจว่า import crew ได้สำเร็จ"""
    from wildfire_prediction.crew import WildfirePredictionCrew
    crew_instance = WildfirePredictionCrew()
    assert crew_instance is not None


def test_tools_import():
    """ตรวจว่า import tools ทั้งหมดได้สำเร็จ"""
    from wildfire_prediction.tools import (
        data_fetcher_tool,
        data_processor_tool,
        ml_trainer_tool,
        map_generator_tool,
        report_formatting_tool,
    )
    assert data_fetcher_tool is not None
    assert data_processor_tool is not None
    assert ml_trainer_tool is not None
    assert map_generator_tool is not None
    assert report_formatting_tool is not None
