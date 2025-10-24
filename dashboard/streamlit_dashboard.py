"""
MLOps Project Structure Dashboard
Streamlit-based validation dashboard for Turkish Music Emotion Recognition project
Team 24 - Tecnológico de Monterrey (ITESM)
"""

import streamlit as st
import subprocess
import os
import sys
from pathlib import Path
import json
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

# Import validator
from validate_cookiecutter import CookieCutterValidator

# Page configuration
st.set_page_config(
    page_title="MLOps Team 24 - Project Validator",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
    }
    
    .success-box {
        background: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        border-radius: 4px;
        margin: 0.5rem 0;
    }
    
    .error-box {
        background: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 1rem;
        border-radius: 4px;
        margin: 0.5rem 0;
    }
    
    .info-box {
        background: #d1ecf1;
        border-left: 4px solid #17a2b8;
        padding: 1rem;
        border-radius: 4px;
        margin: 0.5rem 0;
    }
    
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🎵 MLOps Project Structure Validator</h1>
    <p><b>Turkish Music Emotion Recognition</b></p>
    <p>Team 24 | Tecnológico de Monterrey | MNA 2025</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/music.png", width=80)
    st.header("⚙️ Configuration")
    
    # Source selection
    source = st.radio(
        "📁 Data Source:",
        ["Local Repository", "GitHub Repository"],
        help="Choose where to validate from"
    )
    
    repo_path = None
    
    if source == "Local Repository":
        default_path = str(Path.cwd())
        repo_path = st.text_input(
            "Repository Path:",
            value=default_path,
            help="Full path to your local repository"
        )
        
        # Validate path exists
        if repo_path and not Path(repo_path).exists():
            st.warning("⚠️ Path does not exist!")
            repo_path = None
    
    else:  # GitHub Repository
        repo_url = st.text_input(
            "GitHub URL:",
            placeholder="https://github.com/username/repo",
            help="Full GitHub repository URL"
        )
        
        if st.button("📥 Clone Repository", type="secondary"):
            if repo_url:
                with st.spinner("Cloning repository..."):
                    clone_path = Path("./temp_repo")
                    
                    # Remove if exists
                    if clone_path.exists():
                        import shutil
                        shutil.rmtree(clone_path)
                    
                    try:
                        result = subprocess.run(
                            ["git", "clone", repo_url, str(clone_path)],
                            capture_output=True,
                            text=True,
                            timeout=60
                        )
                        
                        if result.returncode == 0:
                            st.success("✅ Repository cloned successfully!")
                            repo_path = str(clone_path)
                            st.session_state['repo_path'] = repo_path
                        else:
                            st.error(f"❌ Clone failed: {result.stderr}")
                    except subprocess.TimeoutExpired:
                        st.error("❌ Clone timeout (60s)")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
            else:
                st.warning("Please enter a GitHub URL")
        
        # Use previously cloned repo
        if 'repo_path' in st.session_state:
            repo_path = st.session_state['repo_path']
            st.info(f"Using: {repo_path}")
    
    st.divider()
    
    # Team info
    st.markdown("### 👥 Team 24")
    st.markdown("""
    - David Cruz Beltrán
    - Javier Rebull Saucedo  
    - Sandra Cervantes Espinoza
    """)
    
    st.divider()
    
    st.markdown("### 📚 Project Info")
    st.markdown("""
    **Emotion Classes:**
    - 😊 Happy
    - 😢 Sad
    - 😠 Angry
    - 😌 Relax
    """)

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📋 Professional MLOps Structure")
    
    structure_tabs = st.tabs(["📁 Directories", "📄 Files", "🔧 MLOps"])
    
    with structure_tabs[0]:
        st.markdown("""
```
        MLOps_Team24/
        ├── data/
        │   ├── raw/              # Datos originales inmutables
        │   ├── processed/        # Datos limpios y transformados
        │   └── external/         # Fuentes de datos externas
        ├── notebooks/            # Jupyter notebooks (EDA, análisis)
        ├── acoustic_ml/          # 🎵 Paquete Python principal
        │   ├── config.py         # Configuraciones
        │   ├── dataset.py        # Manejo de datos
        │   ├── features.py       # Feature engineering
        │   ├── modeling/         # Entrenamiento y predicción
        │   └── plots.py          # Visualizaciones
        ├── app/                  # 🚀 API FastAPI
        │   ├── api/              # Endpoints
        │   ├── core/             # Config y logging
        │   └── services/         # Servicios de modelo
        ├── models/               # Modelos entrenados
        │   ├── baseline/         # Modelos baseline
        │   └── optimized/        # Modelos optimizados
        ├── reports/              # Reportes de análisis
        │   └── figures/          # Figuras generadas
        ├── references/           # Diccionarios, PDFs, referencias
        └── docs/                 # Documentación del proyecto
```
        """)
    
    with structure_tabs[1]:
        st.markdown("""
        **Archivos Esenciales:**
        - `README.md` - Documentación del proyecto
        - `requirements.txt` - Dependencias Python
        - `pyproject.toml` - Configuración moderna Python
        - `.gitignore` / `.dvcignore` - Reglas Git/DVC
        - `Makefile` - Comandos automatizados
        """)
    
    with structure_tabs[2]:
        st.markdown("""
        **Configuración MLOps Team 24:**
        - `dvc.yaml` - Pipeline de DVC
        - `params.yaml` - Hiperparámetros
        - `.dvc/config` - Configuración DVC remoto
        - `mlruns/` - Experimentos MLflow
        - `docker-compose.yml` - Deployment containers
        """)
with col2:
    st.header("🚀 Actions")
    
    # Validation button
    if st.button("🔍 Run Validation", type="primary", use_container_width=True):
        if repo_path and Path(repo_path).exists():
            with st.spinner("Validating structure..."):
                try:
                    validator = CookieCutterValidator(repo_path)
                    result = validator.validate()
                    
                    # Store in session
                    st.session_state['validation_result'] = result
                    st.session_state['validation_time'] = datetime.now()
                    st.session_state['validated_path'] = repo_path
                    
                    st.success("✅ Validation complete!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Validation error: {str(e)}")
        else:
            st.warning("⚠️ Please select a valid repository path")
    
    # Export button
    if 'validation_result' in st.session_state:
        result = st.session_state['validation_result']
        
        st.download_button(
            label="📥 Download JSON Report",
            data=json.dumps(result, indent=2),
            file_name=f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )
        
        if st.button("🗑️ Clear Results", use_container_width=True):
            del st.session_state['validation_result']
            del st.session_state['validation_time']
            st.rerun()

# Results section
if 'validation_result' in st.session_state:
    st.divider()
    
    result = st.session_state['validation_result']
    summary = result['summary']
    
    # Header with score
    col1, col2 = st.columns([3, 1])
    with col1:
        st.header("📊 Validation Results")
    with col2:
        score = summary['overall']['score']
        if score >= 90:
            st.success(f"🎯 Score: {score}%")
        elif score >= 70:
            st.warning(f"⚠️ Score: {score}%")
        else:
            st.error(f"❌ Score: {score}%")
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📁 Directories",
            value=f"{summary['directories']['present']}/{summary['directories']['total']}",
            delta=f"{summary['directories']['percentage']}%"
        )
    
    with col2:
        st.metric(
            label="📄 Files",
            value=f"{summary['files']['present']}/{summary['files']['total']}",
            delta=f"{summary['files']['percentage']}%"
        )
    
    with col3:
        st.metric(
            label="🔧 MLOps",
            value=f"{summary['mlops']['present']}/{summary['mlops']['total']}",
            delta=f"{summary['mlops']['percentage']}%"
        )
    
    with col4:
        st.metric(
            label="🎯 Overall",
            value=f"{summary['overall']['present']}/{summary['overall']['total']}",
            delta=f"{summary['overall']['score']}%"
        )
    
    # Progress bar
    st.progress(summary['overall']['score'] / 100, text=f"Project Completeness: {summary['overall']['score']}%")
    
    # Visualization
    st.subheader("📈 Completeness Breakdown")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Pie chart
        categories = ['Directories', 'Files', 'MLOps']
        percentages = [
            summary['directories']['percentage'],
            summary['files']['percentage'],
            summary['mlops']['percentage']
        ]
        
        fig = go.Figure(data=[go.Pie(
            labels=categories,
            values=percentages,
            hole=.3,
            marker=dict(colors=['#667eea', '#764ba2', '#f093fb'])
        )])
        
        fig.update_layout(
            title="Completion by Category",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Bar chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Present',
            x=categories,
            y=[summary['directories']['present'], 
               summary['files']['present'],
               summary['mlops']['present']],
            marker_color='#28a745'
        ))
        
        fig.add_trace(go.Bar(
            name='Missing',
            x=categories,
            y=[summary['directories']['missing'],
               summary['files']['missing'],
               summary['mlops']['missing']],
            marker_color='#dc3545'
        ))
        
        fig.update_layout(
            title="Items Status",
            barmode='stack',
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed results
    st.divider()
    st.subheader("🔍 Detailed Results")
    
    detail_tabs = st.tabs(["📁 Directories", "📄 Files", "🔧 MLOps Files", "📋 Optional", "🌳 Tree View"])
    
    with detail_tabs[0]:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### ✅ Present Directories")
            if result['directories']['present']:
                for item in result['directories']['present']:
                    st.markdown(f"- ✓ `{item}`")
            else:
                st.info("None found")
        
        with col2:
            st.markdown("### ❌ Missing Directories")
            if result['directories']['missing']:
                for item in result['directories']['missing']:
                    st.markdown(f"- ✗ `{item}`")
            else:
                st.success("All directories present! 🎉")
    
    with detail_tabs[1]:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### ✅ Present Files")
            if result['files']['present']:
                for item in result['files']['present']:
                    st.markdown(f"- ✓ `{item}`")
            else:
                st.info("None found")
        
        with col2:
            st.markdown("### ❌ Missing Files")
            if result['files']['missing']:
                for item in result['files']['missing']:
                    st.markdown(f"- ✗ `{item}`")
            else:
                st.success("All files present! 🎉")
    
    with detail_tabs[2]:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### ✅ Present MLOps Files")
            if result['mlops']['present']:
                for item in result['mlops']['present']:
                    st.markdown(f"- ✓ `{item}`")
            else:
                st.warning("No MLOps files found")
        
        with col2:
            st.markdown("### ❌ Missing MLOps Files")
            if result['mlops']['missing']:
                for item in result['mlops']['missing']:
                    st.markdown(f"- ✗ `{item}`")
            else:
                st.success("All MLOps files present! 🎉")
    
    with detail_tabs[3]:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### ✅ Present Optional Items")
            if result['optional']['present']:
                for item in result['optional']['present']:
                    st.markdown(f"- ✓ `{item}`")
            else:
                st.info("None found")
        
        with col2:
            st.markdown("### ❌ Missing Optional Items")
            if result['optional']['missing']:
                for item in result['optional']['missing']:
                    st.markdown(f"- ○ `{item}`")
            else:
                st.info("All optional items present")
    
    with detail_tabs[4]:
        st.markdown("### 🌳 Repository Structure")
        try:
            validator = CookieCutterValidator(st.session_state['validated_path'])
            tree = validator.get_tree_structure(max_depth=3)
            st.code(tree, language="text")
        except Exception as e:
            st.error(f"Could not generate tree: {str(e)}")
    
    # Timestamp
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.caption(f"🕐 Last validated: {st.session_state['validation_time'].strftime('%Y-%m-%d %H:%M:%S')}")
    with col2:
        st.caption(f"📂 Path: {st.session_state['validated_path']}")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p><b>MLOps Team 24</b> | Tecnológico de Monterrey (ITESM)</p>
    <p>🎓 Master in Applied Artificial Intelligence (MNA) | 2025</p>
    <p>🎵 Turkish Music Emotion Recognition Project</p>
    <p style='font-size: 0.8em; margin-top: 1rem;'>
        Supervisors: Dr. Gerardo Rodríguez Hernández & Mtro. Ricardo Valdez Hernández
    </p>
</div>
""", unsafe_allow_html=True)
