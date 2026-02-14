"""Wildfire Prediction Crew assembly using CrewAI @CrewBase decorator pattern."""

from crewai import Agent, Crew, LLM, Process, Task
from crewai.project import CrewBase, agent, crew, task

from wildfire_prediction.tools.data_fetcher import data_fetcher_tool
from wildfire_prediction.tools.data_processor import data_processor_tool
from wildfire_prediction.tools.ml_trainer import ml_trainer_tool
from wildfire_prediction.tools.map_generator import map_generator_tool
from wildfire_prediction.tools.report_tool import report_formatting_tool


@CrewBase
class WildfirePredictionCrew:
    """Wildfire Prediction Crew — sequential multi-agent pipeline."""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def data_collector(self) -> Agent:
        """Remote Sensing Data Specialist agent."""
        return Agent(
            config=self.agents_config["data_collector"],
            tools=[data_fetcher_tool],
        )

    @agent
    def data_analyst(self) -> Agent:
        """Geospatial Data Analyst agent."""
        return Agent(
            config=self.agents_config["data_analyst"],
            tools=[data_processor_tool],
        )

    @agent
    def ml_engineer(self) -> Agent:
        """Deep Learning Engineer agent."""
        return Agent(
            config=self.agents_config["ml_engineer"],
            tools=[ml_trainer_tool],
        )

    @agent
    def cartographer(self) -> Agent:
        """GIS Cartographer agent."""
        return Agent(
            config=self.agents_config["cartographer"],
            tools=[map_generator_tool],
        )

    @agent
    def report_writer(self) -> Agent:
        """Academic Research Writer agent."""
        return Agent(
            config=self.agents_config["report_writer"],
            tools=[report_formatting_tool],
        )

    @task
    def collect_and_validate_data(self) -> Task:
        """Data collection and validation task."""
        return Task(config=self.tasks_config["collect_and_validate_data"])

    @task
    def analyze_and_engineer_features(self) -> Task:
        """Feature engineering task."""
        return Task(config=self.tasks_config["analyze_and_engineer_features"])

    @task
    def train_and_evaluate_model(self) -> Task:
        """Model training and evaluation task."""
        return Task(config=self.tasks_config["train_and_evaluate_model"])

    @task
    def generate_risk_maps(self) -> Task:
        """Risk map generation task."""
        return Task(config=self.tasks_config["generate_risk_maps"])

    @task
    def write_research_report(self) -> Task:
        """Research report writing task."""
        return Task(config=self.tasks_config["write_research_report"])

    @crew
    def crew(self) -> Crew:
        """Assemble the Wildfire Prediction Crew."""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
