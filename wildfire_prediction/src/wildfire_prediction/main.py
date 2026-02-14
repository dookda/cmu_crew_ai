"""Entry point for the Wildfire Prediction CrewAI pipeline."""

import time

from dotenv import load_dotenv
load_dotenv()

from wildfire_prediction.crew import WildfirePredictionCrew


def run() -> None:
    """Run the wildfire prediction pipeline with timing measurement."""
    inputs = {
        "province": "Chiang Mai",
        "start_year": "2015",
        "end_year": "2024",
        "data_path": "data/sample_hotspot_data.csv",
    }

    print("=" * 60)
    print("WILDFIRE HOTSPOT PREDICTION PIPELINE")
    print("=" * 60)
    print(f"Province: {inputs['province']}")
    print(f"Period: {inputs['start_year']} - {inputs['end_year']}")
    print(f"Data: {inputs['data_path']}")
    print("=" * 60)

    start_time = time.time()

    crew = WildfirePredictionCrew()
    result = crew.crew().kickoff(inputs=inputs)

    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = elapsed % 60

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print(f"Total time: {minutes}m {seconds:.1f}s")
    print("=" * 60)
    print("\nFinal Result:")
    print(result)


if __name__ == "__main__":
    run()
