"""
🔥 Data Drift Detection Dashboard - Premium UI Version
Turkish Music Emotion Recognition - MLOps Team 24
Streamlit + Plotly + Custom Color Palette
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
import numpy as np


# Page configuration
st.set_page_config(
    page_title="🔥 Data Drift Dashboard | MLOps Team24",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM COLOR PALETTE ====================
colors = {
    'primary': '#17A2A2',      # Verde-teal más vivido
    'bg': '#F0FFFE',           # Fondo blanco ligeramente más vivido
    'border': '#5FB3B3',       # Bordes verde-teal vivido
    'surface': '#B3E5E1',      # Tarjetas verde pastel vivido
    'surface_2': '#D4F4F1',    # Resaltados verde muy claro pero vivido
    'muted': '#7FC4C1',        # Texto secundario verde-teal
    'text_strong': '#1a3a3a',  # Texto fuerte más oscuro
    'text_base': '#2d4a49',    # Texto base oscuro
}

# Colores para gráficos - verdes vividos
chart_colors = ['#17A2A2', '#20B3B3', '#29C4C4', '#32D5D5', '#3BE6E6', '#44F7F7']
impact_colors = {
    'low': '#17A2A2',
    'medium': '#20B3B3',
    'high': '#FF9F43',
    'critical': '#EE5A6F'
}

# ==================== CUSTOM CSS ====================
st.markdown(f"""
<style>
    * {{
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }}
    
    /* Background y texto base */
    body, .main {{
        background: {colors['bg']};
        color: {colors['text_base']};
    }}
    
    /* Sidebar */
    .sidebar .sidebar-content {{
        background: {colors['surface_2']};
    }}
    
    /* Headers */
    h1, h2, h3 {{
        color: {colors['text_strong']};
        font-weight: 700;
        letter-spacing: -0.5px;
    }}
    
    h1 {{
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }}
    
    h2 {{
        font-size: 1.8rem;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        border-bottom: 3px solid {colors['primary']};
        padding-bottom: 0.5rem;
    }}
    
    h3 {{
        font-size: 1.3rem;
        margin-top: 1rem;
        margin-bottom: 0.8rem;
    }}
    
    /* Metric Cards */
    .metric-card {{
        background: linear-gradient(135deg, {colors['surface']} 0%, {colors['surface_2']} 100%);
        border: 1px solid {colors['border']};
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(155, 180, 181, 0.15);
        transition: all 0.3s ease;
    }}
    
    .metric-card:hover {{
        box-shadow: 0 8px 20px rgba(155, 180, 181, 0.25);
        transform: translateY(-2px);
    }}
    
    .metric-value {{
        font-size: 2.2rem;
        font-weight: 700;
        color: {colors['primary']};
        margin: 12px 0;
    }}
    
    .metric-label {{
        font-size: 0.95rem;
        color: {colors['muted']};
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    
    /* Alert Boxes */
    .alert-critical {{
        background: linear-gradient(135deg, #f5e6e3 0%, #faf0ed 100%);
        border-left: 4px solid #D67A6B;
        border-radius: 8px;
        padding: 16px;
        margin: 16px 0;
        color: {colors['text_strong']};
    }}
    
    .alert-warning {{
        background: linear-gradient(135deg, #fef3e8 0%, #fffaf4 100%);
        border-left: 4px solid #E5A89B;
        border-radius: 8px;
        padding: 16px;
        margin: 16px 0;
        color: {colors['text_strong']};
    }}
    
    .alert-info {{
        background: linear-gradient(135deg, {colors['surface']} 0%, {colors['surface_2']} 100%);
        border-left: 4px solid {colors['primary']};
        border-radius: 8px;
        padding: 16px;
        margin: 16px 0;
        color: {colors['text_strong']};
    }}
    
    .alert-title {{
        font-weight: 700;
        margin-bottom: 6px;
        font-size: 1.05rem;
    }}
    
    /* Tablas */
    .dataframe {{
        border: 1px solid {colors['border']};
        border-radius: 8px;
        overflow: hidden;
    }}
    
    .dataframe th {{
        background: {colors['surface']};
        color: {colors['text_strong']};
        border-bottom: 2px solid {colors['border']};
        padding: 12px;
        font-weight: 700;
    }}
    
    .dataframe td {{
        border-bottom: 1px solid {colors['muted']};
        padding: 12px;
        color: {colors['text_base']};
    }}
    
    .dataframe tr:hover {{
        background: {colors['surface_2']};
    }}
    
    /* Buttons y Selectores */
    .stButton > button {{
        background: {colors['primary']};
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 600;
        transition: all 0.3s ease;
    }}
    
    .stButton > button:hover {{
        background: {colors['text_strong']};
        box-shadow: 0 4px 12px rgba(155, 180, 181, 0.3);
    }}
    
    /* Dividers */
    hr {{
        border: none;
        border-top: 2px solid {colors['border']};
        margin: 2rem 0;
    }}
    
    /* Footer */
    .footer {{
        text-align: center;
        color: {colors['muted']};
        font-size: 0.9rem;
        padding: 24px;
        border-top: 1px solid {colors['border']};
        margin-top: 3rem;
    }}
</style>
""", unsafe_allow_html=True)


# ==================== HELPER FUNCTIONS ====================
@st.cache_data
def load_json(filepath):
    """Load JSON file with error handling"""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading {filepath}: {e}")
        return None


def load_scenario_data():
    """Load all drift scenario data"""
    base_path = Path(__file__).parent
    
    scenarios = {
        'baseline': load_json(base_path / 'drift_baseline.json'),
        'mean_shift': load_json(base_path / 'drift_mean_shift.json'),
        'variance_change': load_json(base_path / 'drift_variance_change.json'),
        'combined_drift': load_json(base_path / 'drift_combined_drift.json'),
    }
    
    metadata = load_json(base_path / 'drift_scenarios_metadata.json')
    executive_summary = load_json(base_path / 'drift_executive_summary.json')
    
    return scenarios, metadata, executive_summary


def create_class_distribution_chart(scenarios):
    """Create beautiful class distribution comparison chart"""
    fig = go.Figure()
    
    scenario_names = {
        'baseline': '📊 Baseline',
        'mean_shift': '➡️ Mean Shift',
        'variance_change': '📈 Variance Change',
        'combined_drift': '⚠️ Combined Drift'
    }
    
    colors_list = ['#17A2A2', '#20B3B3', '#29C4C4', '#32D5D5']
    
    for idx, (scenario_key, scenario_name) in enumerate(scenario_names.items()):
        if scenarios[scenario_key]:
            class_dist = scenarios[scenario_key]['metrics']['class_dist']
            classes = sorted([int(k) for k in class_dist.keys()])
            counts = [class_dist[str(c)] for c in classes]
            
            emotion_labels = {0: 'Angry 😠', 1: 'Happy 😊', 2: 'Relax 😌', 3: 'Sad 😢'}
            x_labels = [f"{emotion_labels.get(c, f'Class {c}')}" for c in classes]
            
            fig.add_trace(go.Bar(
                x=x_labels,
                y=counts,
                name=scenario_name,
                marker=dict(color=colors_list[idx], line=dict(color=colors['border'], width=1)),
                text=counts,
                textposition='auto',
                textfont=dict(color=colors['text_strong'], size=11),
                hovertemplate='<b>%{x}</b><br>Muestras: %{y}<extra></extra>',
                opacity=0.85
            ))
    
    fig.update_layout(
        title=dict(
            text='<b>Distribución de Clases de Emoción</b>',
            font=dict(size=18, color=colors['text_strong']),
            x=0.5,
            xanchor='center'
        ),
        xaxis_title='<b>Emociones</b>',
        yaxis_title='<b>Cantidad de Muestras</b>',
        barmode='group',
        hovermode='closest',
        height=500,
        plot_bgcolor=colors['bg'],
        paper_bgcolor=colors['bg'],
        font=dict(size=11, color=colors['text_base']),
        showlegend=True,
        legend=dict(bgcolor='rgba(255,255,255,0)', bordercolor=colors['border'], borderwidth=1),
        xaxis=dict(showgrid=False, linecolor=colors['border']),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor=colors['muted'], linecolor=colors['border'])
    )
    
    return fig


def create_mean_prediction_chart(executive_summary):
    """Create mean prediction comparison chart with annotations"""
    scenarios_list = ['Baseline', 'Mean Shift', 'Variance Change', 'Combined Drift']
    mean_predictions = [
        executive_summary['baseline_prediction_mean'],
        executive_summary['scenarios']['mean_shift']['mean_prediction'],
        executive_summary['scenarios']['variance_change']['mean_prediction'],
        executive_summary['scenarios']['combined_drift']['mean_prediction'],
    ]
    
    changes = [0.0] + [
        executive_summary['scenarios']['mean_shift']['change_percent'],
        executive_summary['scenarios']['variance_change']['change_percent'],
        executive_summary['scenarios']['combined_drift']['change_percent'],
    ]
    
    # Color based on impact
    bar_colors = [
        colors['primary'],
        impact_colors['low'] if changes[1] < 1 else impact_colors['high'],
        impact_colors['high'] if changes[2] >= 3 else impact_colors['medium'],
        impact_colors['critical'] if changes[3] >= 4 else impact_colors['high'],
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=scenarios_list,
        y=mean_predictions,
        marker=dict(
            color=bar_colors,
            line=dict(color=colors['border'], width=2)
        ),
        text=[f"<b>{mp:.3f}</b><br>({ch:+.2f}%)" for mp, ch in zip(mean_predictions, changes)],
        textposition='outside',
        textfont=dict(color=colors['text_strong'], size=12),
        hovertemplate='<b>%{x}</b><br>Media: %{y:.4f}<extra></extra>',
        name='Media de Predicción',
        opacity=0.85
    ))
    
    fig.update_layout(
        title=dict(
            text='<b>Cambio en la Media de Predicciones</b>',
            font=dict(size=18, color=colors['text_strong']),
            x=0.5,
            xanchor='center'
        ),
        xaxis_title='<b>Escenario</b>',
        yaxis_title='<b>Valor de Media de Predicción</b>',
        height=450,
        plot_bgcolor=colors['bg'],
        paper_bgcolor=colors['bg'],
        font=dict(size=12, color=colors['text_base']),
        showlegend=False,
        xaxis=dict(showgrid=False, linecolor=colors['border']),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor=colors['muted'], linecolor=colors['border']),
        margin=dict(t=80, b=60, l=60, r=60)
    )
    
    return fig


def create_recommendation_table(executive_summary):
    """Create recommendations dataframe"""
    data = {
        'Escenario': ['Mean Shift', 'Variance Change', 'Combined Drift'],
        'Nivel de Impacto': [
            executive_summary['scenarios']['mean_shift']['impact_level'],
            executive_summary['scenarios']['variance_change']['impact_level'],
            executive_summary['scenarios']['combined_drift']['impact_level'],
        ],
        'Cambio (%)': [
            f"{executive_summary['scenarios']['mean_shift']['change_percent']:.2f}%",
            f"{executive_summary['scenarios']['variance_change']['change_percent']:.2f}%",
            f"{executive_summary['scenarios']['combined_drift']['change_percent']:.2f}%",
        ],
        'Recomendación': [
            executive_summary['scenarios']['mean_shift']['recommendation'],
            executive_summary['scenarios']['variance_change']['recommendation'],
            executive_summary['scenarios']['combined_drift']['recommendation'],
        ]
    }
    return pd.DataFrame(data)


def create_metadata_table(metadata):
    """Create metadata information table"""
    data = {
        'Aspecto': [
            'Muestras de Entrenamiento',
            'Features/Características',
            'Generado en',
            'Ruta de Datos'
        ],
        'Valor': [
            metadata['training_data']['rows'],
            metadata['training_data']['columns'],
            metadata['generated_at'][:19],
            metadata['training_data']['path']
        ]
    }
    return pd.DataFrame(data)


def create_impact_summary_cards(executive_summary):
    """Create three summary cards for quick insights"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">📊 Media Baseline</div>
            <div class="metric-value">{executive_summary['baseline_prediction_mean']:.4f}</div>
            <div class="metric-label">Valor de referencia</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        highest_impact = max(
            executive_summary['scenarios']['mean_shift']['change_percent'],
            executive_summary['scenarios']['variance_change']['change_percent'],
            executive_summary['scenarios']['combined_drift']['change_percent'],
        )
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">⚡ Mayor Impacto</div>
            <div class="metric-value">{highest_impact:.2f}%</div>
            <div class="metric-label">Drift detectado</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        critical_count = sum(1 for s in executive_summary['scenarios'].values() 
                           if 'HIGH' in s['impact_level'] or 'CRITICAL' in s.get('impact_level', ''))
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">🔴 Escenarios Críticos</div>
            <div class="metric-value">{critical_count}/3</div>
            <div class="metric-label">Requieren atención</div>
        </div>
        """, unsafe_allow_html=True)


def create_alert_box(alert_type, title, message):
    """Create styled alert box"""
    if alert_type == 'critical':
        st.markdown(f"""
        <div class="alert-critical">
            <div class="alert-title">🔴 {title}</div>
            <p>{message}</p>
        </div>
        """, unsafe_allow_html=True)
    elif alert_type == 'warning':
        st.markdown(f"""
        <div class="alert-warning">
            <div class="alert-title">⚠️ {title}</div>
            <p>{message}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="alert-info">
            <div class="alert-title">ℹ️ {title}</div>
            <p>{message}</p>
        </div>
        """, unsafe_allow_html=True)


# ==================== MAIN DASHBOARD ====================
def main():
    # Header
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1>🔥 Data Drift Detection Dashboard</h1>
        <p style="font-size: 1.1rem; color: #9BB4B5; margin-top: 0.5rem;">
            Turkish Music Emotion Recognition - MLOps Team 24
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Load data
    scenarios, metadata, executive_summary = load_scenario_data()
    
    if not all([scenarios, metadata, executive_summary]):
        st.error("❌ Error cargando datos del dashboard. Verifica las rutas de los archivos.")
        return
    
    # Sidebar
    st.sidebar.markdown("## 🎛️ Controles del Dashboard")
    view_mode = st.sidebar.radio(
        "Selecciona vista:",
        ["📊 Resumen Ejecutivo", "📈 Análisis Detallado", "🔍 Comparación de Escenarios", "ℹ️ Metadatos"],
        label_visibility="collapsed"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 Información del Proyecto")
    st.sidebar.info("""
    **Team:** MLOps Team 24  
    **Proyecto:** Turkish Music Emotion Recognition  
    **Fase:** 3 - Production Deployment  
    **Estado:** ✅ Monitoring Activo
    """)
    
    # ==================== EXECUTIVE SUMMARY ====================
    if view_mode == "📊 Resumen Ejecutivo":
        st.subheader("📊 Resumen Ejecutivo de Data Drift")
        
        # Impact Summary Cards
        create_impact_summary_cards(executive_summary)
        
        st.markdown("---")
        
        # Charts
        st.subheader("📉 Visualización de Impacto")
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(create_mean_prediction_chart(executive_summary), use_container_width=True)
        
        with col2:
            st.plotly_chart(create_class_distribution_chart(scenarios), use_container_width=True)
        
        st.markdown("---")
        
        # Recommendations
        st.subheader("📋 Recomendaciones y Acciones")
        recommendations_df = create_recommendation_table(executive_summary)
        st.dataframe(recommendations_df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # Alerts
        st.subheader("⚠️ Alertas e Insights")
        
        if executive_summary['scenarios']['variance_change']['change_percent'] >= 3:
            create_alert_box(
                'warning',
                'Variance Drift Detectado',
                'El escenario de cambio de varianza muestra un impacto del 3.37% en las predicciones. El modelo es vulnerable a mayor ruido en las características de entrada.'
            )
        
        if executive_summary['scenarios']['combined_drift']['change_percent'] >= 4:
            create_alert_box(
                'critical',
                'Combined Drift Crítico',
                'El drift multi-dimensional (mean shift + variance + outliers) causa un impacto del 4.33%. Se requiere re-entrenamiento inmediato del modelo.'
            )
        
        if executive_summary['scenarios']['mean_shift']['change_percent'] < 1:
            create_alert_box(
                'info',
                'Mean Shift Robusto',
                'El modelo muestra excelente robustez ante desplazamientos de media sistemáticos (+30%). Indica buena generalización a ciertos tipos de cambios.'
            )
    
    # ==================== DETAILED ANALYSIS ====================
    elif view_mode == "📈 Análisis Detallado":
        st.subheader("📈 Análisis Profundo de Escenarios")
        
        selected_scenario = st.selectbox(
            "Selecciona un escenario para analizar:",
            ["Baseline", "Mean Shift", "Variance Change", "Combined Drift"],
            format_func=lambda x: f"{'📊' if x == 'Baseline' else '➡️' if x == 'Mean Shift' else '📈' if x == 'Variance Change' else '⚠️'} {x}"
        )
        
        scenario_key = {
            "Baseline": "baseline",
            "Mean Shift": "mean_shift",
            "Variance Change": "variance_change",
            "Combined Drift": "combined_drift"
        }[selected_scenario]
        
        scenario_data = scenarios[scenario_key]
        
        if scenario_data:
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📊 Total Muestras", len(scenario_data['metrics']['predictions']))
            with col2:
                st.metric("🎯 Media", f"{scenario_data['metrics']['mean_pred']:.4f}")
            with col3:
                st.metric("📍 Clases", len(scenario_data['metrics']['unique_classes']))
            with col4:
                st.metric("⏱️ Fecha", scenario_data['timestamp'][:10])
            
            st.markdown("---")
            
            # Class distribution
            st.subheader("📊 Distribución de Emociones")
            class_dist = scenario_data['metrics']['class_dist']
            classes = sorted([int(k) for k in class_dist.keys()])
            emotion_labels = {0: 'Angry 😠', 1: 'Happy 😊', 2: 'Relax 😌', 3: 'Sad 😢'}
            
            chart_data = {
                'Emoción': [emotion_labels.get(c, f'Class {c}') for c in classes],
                'Cantidad': [class_dist[str(c)] for c in classes],
                'Porcentaje': [f"{(class_dist[str(c)] / len(scenario_data['metrics']['predictions']) * 100):.1f}%" 
                             for c in classes]
            }
            
            fig = px.bar(
                chart_data,
                x='Emoción',
                y='Cantidad',
                color='Emoción',
                text='Porcentaje',
                title=f"Distribución de Clases - {selected_scenario}",
                height=400,
                color_discrete_sequence=['#17A2A2', '#20B3B3', '#29C4C4', '#32D5D5']
            )
            fig.update_traces(textposition='auto', marker=dict(line=dict(color=colors['border'], width=1)))
            fig.update_layout(plot_bgcolor=colors['bg'], paper_bgcolor=colors['bg'])
            st.plotly_chart(fig, use_container_width=True)
            
            # Prediction histogram
            st.subheader("📈 Distribución de Predicciones")
            predictions = scenario_data['metrics']['predictions']
            
            fig = go.Figure(data=[
                go.Histogram(
                    x=predictions,
                    nbinsx=4,
                    marker=dict(color='#17A2A2', line=dict(color=colors['border'], width=1)),
                    hovertemplate='<b>Clase %{x}</b><br>Frecuencia: %{y}<extra></extra>'
                )
            ])
            fig.update_layout(
                title=f"Histograma de Predicciones - {selected_scenario}",
                xaxis_title="Clase Predicha",
                yaxis_title="Frecuencia",
                height=400,
                plot_bgcolor=colors['bg'],
                paper_bgcolor=colors['bg'],
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridwidth=1, gridcolor=colors['muted'])
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # ==================== SCENARIO COMPARISON ====================
    elif view_mode == "🔍 Comparación de Escenarios":
        st.subheader("🔍 Comparación Multi-Escenario")
        
        st.plotly_chart(create_class_distribution_chart(scenarios), use_container_width=True)
        st.plotly_chart(create_mean_prediction_chart(executive_summary), use_container_width=True)
        
        st.markdown("---")
        st.subheader("📊 Tabla Resumen de Escenarios")
        
        comparison_data = []
        scenario_info = {
            'baseline': ('Baseline', '✅ Sin drift aplicado'),
            'mean_shift': ('Mean Shift', '➡️ +30% en todas las medias'),
            'variance_change': ('Variance Change', '📈 Varianza ×1.5'),
            'combined_drift': ('Combined Drift', '⚠️ Mean +20% + Var ×1.3 + outliers'),
        }
        
        for key, (name, desc) in scenario_info.items():
            if scenarios[key]:
                comparison_data.append({
                    'Escenario': name,
                    'Descripción': desc,
                    'Media Pred': f"{scenarios[key]['metrics']['mean_pred']:.4f}",
                    'Total Muestras': len(scenarios[key]['metrics']['predictions']),
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    # ==================== METADATA ====================
    elif view_mode == "ℹ️ Metadatos":
        st.subheader("ℹ️ Metadatos del Dataset")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📋 Información del Entrenamiento")
            metadata_df = create_metadata_table(metadata)
            st.dataframe(metadata_df, use_container_width=True, hide_index=True)
        
        with col2:
            st.subheader("🎯 Escenarios de Drift")
            scenarios_list = []
            for scenario_key, scenario_info in metadata['scenarios'].items():
                scenarios_list.append({
                    'Escenario': scenario_key.replace('_', ' ').title(),
                    'Descripción': scenario_info['description'],
                    'Impacto': scenario_info['impact'],
                })
            scenarios_df = pd.DataFrame(scenarios_list)
            st.dataframe(scenarios_df, use_container_width=True, hide_index=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div class="footer">
        <p><b>🔥 MLOps Team 24 - Turkish Music Emotion Recognition</b></p>
        <p>Data Drift Detection & Monitoring Dashboard</p>
        <p style="font-size: 0.8rem; margin-top: 1rem; color: #D3DFE1;">
            Phase 3 • Production Deployment • Real-time Monitoring
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
