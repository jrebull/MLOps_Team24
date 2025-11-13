#!/usr/bin/env python3
"""
🔥 GENERADOR DE DATOS CON DRIFT SIMULADO
==========================================
Genera 3 escenarios de data drift para demostración:
  1. Desplazamiento de medias (Mean Shift)
  2. Cambio de varianza (Variance Change)
  3. Combinado (Mean + Variance + Outliers)

Uso:
    python drift/generate_drift_data.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime

class DriftDataGenerator:
    def __init__(self, train_path: str = "data/processed/X_train.csv"):
        self.train_path = Path(train_path)
        self.train_df = None
        self.drift_scenarios = {}
        
    def load_training_data(self):
        """Carga datos de entrenamiento como referencia"""
        self.train_df = pd.read_csv(self.train_path)
        print(f"✓ Datos de entrenamiento cargados: {self.train_df.shape}")
        print(f"  Columnas: {self.train_df.shape[1]}")
        print(f"  Filas: {self.train_df.shape[0]}")
        return self.train_df
    
    def scenario_1_mean_shift(self, shift_factor=0.3):
        """
        Escenario 1: Desplazamiento de medias
        Simula cambios en la distribución sin perder varianza
        (ej: micrófono con calibración diferente)
        """
        print("\n📊 Generando Escenario 1: Mean Shift (+30%)...")
        data = self.train_df.copy()
        
        # Desplazar medias de todas las features
        for col in data.columns:
            mean_val = data[col].mean()
            data[col] = data[col] + (mean_val * shift_factor)
        
        self.drift_scenarios['mean_shift'] = {
            'data': data,
            'description': 'Mean Shift: +30% desplazamiento en todas las medias',
            'impact': 'Alto - Cambio sistemático en todas las features',
            'cause': 'Posible cambio en equipamiento o condiciones de captura'
        }
        print(f"  ✓ {len(data)} muestras generadas")
        return data
    
    def scenario_2_variance_change(self, variance_factor=1.5):
        """
        Escenario 2: Cambio de varianza
        Aumenta la dispersión de los datos (datos más ruidosos)
        (ej: micrófono con más ruido ambiental)
        """
        print("\n📊 Generando Escenario 2: Variance Change (×1.5)...")
        data = self.train_df.copy()
        
        # Aumentar varianza: (x - mean) * factor + mean
        for col in data.columns:
            mean_val = data[col].mean()
            std_val = data[col].std()
            # Reescalar manteniendo la media pero aumentando std
            data[col] = ((data[col] - mean_val) * variance_factor) + mean_val
        
        self.drift_scenarios['variance_change'] = {
            'data': data,
            'description': f'Variance Change: Aumento de varianza ×{variance_factor}',
            'impact': 'Medio-Alto - Mayor ruido en features',
            'cause': 'Mayor variabilidad en condiciones de captura'
        }
        print(f"  ✓ {len(data)} muestras generadas")
        return data
    
    def scenario_3_combined_drift(self, mean_shift=0.2, variance_factor=1.3, outlier_pct=0.05):
        """
        Escenario 3: Drift Combinado
        Combina: desplazamiento, varianza y outliers
        (ej: cambio gradual en equipamiento + anomalías)
        """
        print("\n📊 Generando Escenario 3: Combined Drift (Mean + Variance + Outliers)...")
        data = self.train_df.copy()
        
        # 1. Desplazar medias
        for col in data.columns:
            mean_val = data[col].mean()
            data[col] = data[col] + (mean_val * mean_shift)
        
        # 2. Aumentar varianza
        for col in data.columns:
            mean_val = data[col].mean()
            data[col] = ((data[col] - mean_val) * variance_factor) + mean_val
        
        # 3. Agregar outliers
        n_outliers = max(1, int(len(data) * outlier_pct))
        outlier_indices = np.random.choice(len(data), n_outliers, replace=False)
        for idx in outlier_indices:
            for col in data.columns:
                # Outliers extremos (3 sigma away)
                std_val = data[col].std()
                data.loc[idx, col] = data[col].mean() + np.random.choice([-3, 3]) * std_val
        
        self.drift_scenarios['combined_drift'] = {
            'data': data,
            'description': f'Combined Drift: Mean +{mean_shift*100:.0f}% + Var ×{variance_factor} + {n_outliers} outliers',
            'impact': 'Crítico - Cambio multi-dimensional',
            'cause': 'Degradación significativa en equipamiento/condiciones'
        }
        print(f"  ✓ {len(data)} muestras generadas (+{n_outliers} outliers)")
        return data
    
    def generate_all_scenarios(self):
        """Genera todos los escenarios"""
        print("\n" + "="*70)
        print("🔥 GENERANDO ESCENARIOS DE DATA DRIFT")
        print("="*70)
        
        self.scenario_1_mean_shift()
        self.scenario_2_variance_change()
        self.scenario_3_combined_drift()
        
        return self.drift_scenarios
    
    def save_combined_drift_data(self, output_path: str = "data/processed/data_drift_prod.csv"):
        """
        Guarda los 3 escenarios combinados en un CSV
        Para análisis de drift con Evidently
        """
        output_path = Path(output_path)
        
        # Combinar todos los escenarios
        all_data = []
        
        # Agregar datos originales con etiqueta
        train_data = self.train_df.copy()
        train_data['__drift_scenario__'] = 'baseline'
        all_data.append(train_data)
        
        # Agregar cada escenario
        for scenario_name, scenario_info in self.drift_scenarios.items():
            scenario_data = scenario_info['data'].copy()
            scenario_data['__drift_scenario__'] = scenario_name
            all_data.append(scenario_data)
        
        # Combinar y guardar
        combined_df = pd.concat(all_data, ignore_index=True)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined_df.to_csv(output_path, index=False)
        
        print(f"\n✓ Datos de drift guardados en: {output_path}")
        print(f"  Total muestras: {len(combined_df)}")
        print(f"  Escenarios: {combined_df['__drift_scenario__'].unique().tolist()}")
        
        return combined_df
    
    def save_metadata(self, metadata_path: str = "reports/drift/drift_scenarios_metadata.json"):
        """Guarda metadatos sobre los escenarios generados"""
        metadata_path = Path(metadata_path)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        
        metadata = {
            'generated_at': datetime.now().isoformat(),
            'training_data': {
                'path': str(self.train_path),
                'shape': self.train_df.shape,
                'rows': self.train_df.shape[0],
                'columns': self.train_df.shape[1]
            },
            'scenarios': {}
        }
        
        for scenario_name, scenario_info in self.drift_scenarios.items():
            metadata['scenarios'][scenario_name] = {
                'description': scenario_info['description'],
                'impact': scenario_info['impact'],
                'cause': scenario_info['cause'],
                'samples': len(scenario_info['data'])
            }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✓ Metadatos guardados en: {metadata_path}")
        return metadata
    
    def print_summary(self):
        """Imprime resumen de los escenarios"""
        print("\n" + "="*70)
        print("📋 RESUMEN DE ESCENARIOS GENERADOS")
        print("="*70)
        
        for scenario_name, scenario_info in self.drift_scenarios.items():
            print(f"\n🔹 {scenario_name.upper()}")
            print(f"   Descripción: {scenario_info['description']}")
            print(f"   Impacto: {scenario_info['impact']}")
            print(f"   Causa: {scenario_info['cause']}")
            print(f"   Muestras: {len(scenario_info['data'])}")
        
        print("\n" + "="*70)

def main():
    """Ejecuta la generación de datos con drift"""
    generator = DriftDataGenerator()
    
    # 1. Cargar datos
    generator.load_training_data()
    
    # 2. Generar escenarios
    generator.generate_all_scenarios()
    
    # 3. Guardar datos combinados
    generator.save_combined_drift_data()
    
    # 4. Guardar metadatos
    generator.save_metadata()
    
    # 5. Imprimir resumen
    generator.print_summary()
    
    print("\n✅ Script completado exitosamente")
    print("📌 Próximo paso: python drift/run_drift.py")

if __name__ == "__main__":
    main()
