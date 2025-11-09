import json
from pathlib import Path

class DriftReporter:
    @staticmethod
    def load_report(json_path: Path):
        with open(json_path, "r") as f:
            return json.load(f)

    @staticmethod
    def print_summary(json_path: Path):
        data = DriftReporter.load_report(json_path)
        print(json.dumps(data, indent=2))