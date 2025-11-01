
## [2.0.0] - 2025-11-01

### Added
- Enhanced feature engineering module (`feature_engineering.py`)
- 16 new derived features based on Cohen's d analysis
- Automated retraining script with impact analysis

### Changed
- MLflow run_id: eb05c7698f12499b86ed35ca6efc15a7 → 4b2f54ba46ed4e1d8500da915cf05ceb
- Angry accuracy improved: 82.8% → 86.21% (+3.41%)

### Trade-offs
- Overall test accuracy: 84.30% → 82.64% (-1.66%)
- Sad accuracy declined: 77.4% → 65% (-12.4%)

### Methodology
- Systematic analysis with 4 specialized scripts
- Cohen's d-based feature selection
- MLOps best practices: versioning, reproducibility, documentation

