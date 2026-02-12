"""Basic tests for the Wildfire Prediction Crew."""

import os

import pytest


def test_sample_data_exists():
    """Verify sample data file exists."""
    assert os.path.exists("data/sample_hotspot_data.csv")


def test_crew_import():
    """Verify crew can be imported."""
    from wildfire_prediction.crew import WildfirePredictionCrew
    crew_instance = WildfirePredictionCrew()
    assert crew_instance is not None


def test_tools_import():
    """Verify all tools can be imported."""
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
