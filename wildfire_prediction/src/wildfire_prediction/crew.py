"""ประกอบทีม Crew สำหรับพยากรณ์ไฟป่า ใช้ CrewAI @CrewBase decorator pattern"""

from crewai import Agent, Crew, LLM, Process, Task
from crewai.project import CrewBase, agent, crew, task

from wildfire_prediction.tools.data_fetcher import data_fetcher_tool
from wildfire_prediction.tools.data_processor import data_processor_tool
from wildfire_prediction.tools.ml_trainer import ml_trainer_tool
from wildfire_prediction.tools.map_generator import map_generator_tool
from wildfire_prediction.tools.report_tool import report_formatting_tool


@CrewBase
class WildfirePredictionCrew:
    """ทีมพยากรณ์ไฟป่า — สายพาน multi-agent แบบ sequential"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    # ===== กำหนด Agents (ตัวแทน AI) =====

    @agent
    def data_collector(self) -> Agent:
        """Agent ผู้เชี่ยวชาญข้อมูลดาวเทียม — โหลดและตรวจสอบคุณภาพข้อมูล"""
        return Agent(
            config=self.agents_config["data_collector"],
            tools=[data_fetcher_tool],
        )

    @agent
    def data_analyst(self) -> Agent:
        """Agent นักวิเคราะห์ข้อมูลเชิงพื้นที่ — สร้าง features และวิเคราะห์ patterns"""
        return Agent(
            config=self.agents_config["data_analyst"],
            tools=[data_processor_tool],
        )

    @agent
    def ml_engineer(self) -> Agent:
        """Agent วิศวกร Deep Learning — ฝึกโมเดล LSTM และ baseline"""
        return Agent(
            config=self.agents_config["ml_engineer"],
            tools=[ml_trainer_tool],
        )

    @agent
    def cartographer(self) -> Agent:
        """Agent นักทำแผนที่ GIS — สร้างแผนที่ความเสี่ยงไฟป่า"""
        return Agent(
            config=self.agents_config["cartographer"],
            tools=[map_generator_tool],
        )

    @agent
    def report_writer(self) -> Agent:
        """Agent นักเขียนงานวิจัย — เขียนรายงานวิจัยฉบับสมบูรณ์"""
        return Agent(
            config=self.agents_config["report_writer"],
            tools=[report_formatting_tool],
        )

    # ===== กำหนด Tasks (งานที่ต้องทำ) =====

    @task
    def collect_and_validate_data(self) -> Task:
        """งานที่ 1: รวบรวมและตรวจสอบคุณภาพข้อมูล"""
        return Task(config=self.tasks_config["collect_and_validate_data"])

    @task
    def analyze_and_engineer_features(self) -> Task:
        """งานที่ 2: วิเคราะห์ข้อมูลและสร้าง features"""
        return Task(config=self.tasks_config["analyze_and_engineer_features"])

    @task
    def train_and_evaluate_model(self) -> Task:
        """งานที่ 3: ฝึกและประเมินโมเดล ML"""
        return Task(config=self.tasks_config["train_and_evaluate_model"])

    @task
    def generate_risk_maps(self) -> Task:
        """งานที่ 4: สร้างแผนที่ความเสี่ยงไฟป่า"""
        return Task(config=self.tasks_config["generate_risk_maps"])

    @task
    def write_research_report(self) -> Task:
        """งานที่ 5: เขียนรายงานวิจัย"""
        return Task(config=self.tasks_config["write_research_report"])

    # ===== ประกอบ Crew (ทีม) =====

    @crew
    def crew(self) -> Crew:
        """ประกอบทีมพยากรณ์ไฟป่า — รันแบบ sequential (ทีละงานตามลำดับ)"""
        return Crew(
            agents=self.agents,    # Agent ทั้ง 5 ตัว
            tasks=self.tasks,      # งานทั้ง 5 งาน
            process=Process.sequential,  # ทำงานตามลำดับ
            verbose=True,          # แสดงรายละเอียดระหว่างทำงาน
        )
