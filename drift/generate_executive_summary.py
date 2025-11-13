#!/usr/bin/env python3
import json
from pathlib import Path

baseline_mean = 1.475177304964539
scenarios = {
    'mean_shift': 1.475177304964539,
    'variance_change': 1.524822695035461,
    'combined_drift': 1.5390070921985815
}

summary = {
    'title': '🔥 Executive Summary: Data Drift Analysis',
    'baseline_prediction_mean': baseline_mean,
    'scenarios': {}
}

for scenario, mean_pred in scenarios.items():
    change_pct = ((mean_pred - baseline_mean) / baseline_mean) * 100
    
    if change_pct == 0:
        impact = 'LOW - No impact detected'
    elif abs(change_pct) < 3:
        impact = 'MEDIUM - Minor distribution shift'
    elif abs(change_pct) < 5:
        impact = 'HIGH - Significant drift'
    else:
        impact = 'CRITICAL - Major performance degradation'
    
    summary['scenarios'][scenario] = {
        'mean_prediction': mean_pred,
        'change_percent': round(change_pct, 2),
        'impact_level': impact,
        'recommendation': 'Monitor' if change_pct < 3 else 'Investigate' if change_pct < 5 else 'Retrain'
    }

output = Path("reports/drift/drift_executive_summary.json")
output.parent.mkdir(parents=True, exist_ok=True)

with open(output, 'w') as f:
    json.dump(summary, f, indent=2)

print("✅ Resumen ejecutivo generado")
print(json.dumps(summary, indent=2))
