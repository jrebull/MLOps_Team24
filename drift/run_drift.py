from .detect_drift import DriftDetector
from .report_drift import DriftReporter
from .drift_config import DriftConfig

if __name__ == "__main__":
    detector = DriftDetector()
    detector.generate_report()

    reporter = DriftReporter()
    reporter.print_summary(DriftConfig.DRIFT_JSON_REPORT)
