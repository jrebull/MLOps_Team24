"""
MLOps Project Structure Dashboard
Streamlit-based validation dashboard for Turkish Music Emotion Recognition project
Actualizado para Fase 2 - Post Cleanup
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

# Add current directory to path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

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
        border-radius: 8px;
        margin: 1rem 0;
    }
    .warning-box {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .info-box {
        background: #d1ecf1;
        border-left: 4px solid #17a2b8;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🎵 MLOps Team 24 - Turkish Music Emotion Recognition</h1>
    <p>Cookiecutter Data Science Structure Validator | Fase 2 - Post Cleanup</p>
    <p><strong>Tecnológico de Monterrey (ITESM)</strong></p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://iili.io/Kw90kmB.png", width=80)
    st.markdown("### 📊 Dashboard Options")
    
    repo_path = st.text_input("Repository Path", value=".", help="Root path of the repository")
    
    if st.button("🔄 Run Validation", type="primary"):
        st.session_state.run_validation = True
    
    st.markdown("---")
    st.markdown("### 👥 Team 24")
    st.markdown("""
    - **David Cruz Beltrán**  
      *Software Engineer*
    - **Javier Augusto Rebull**  
      *SRE / Data Engineer*
    - **Sandra Cervantes**  
      *ML Engineer*
    """)
    
    st.markdown("---")
    st.markdown("### 📅 Project Info")
    st.markdown(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    st.markdown("**Phase:** Fase 2 - Completed ✅")

# Main content
if 'run_validation' not in st.session_state:
    st.session_state.run_validation = False

if st.session_state.run_validation or st.sidebar.button("▶️ Auto-run"):
    
    with st.spinner("🔍 Validating project structure..."):
        try:
            validator = CookieCutterValidator(repo_path)
            results = validator.validate()
            
            # Overall Score
            overall_score = results['summary']['overall_score']
            
            st.markdown("## 📊 Validation Results")
            
            # Score gauge
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=overall_score,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Overall Compliance Score", 'font': {'size': 24}},
                    delta={'reference': 90, 'increasing': {'color': "green"}},
                    gauge={
                        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                        'bar': {'color': "#667eea"},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [0, 50], 'color': '#ffcccc'},
                            {'range': [50, 80], 'color': '#fff9c4'},
                            {'range': [80, 100], 'color': '#d4edda'}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            # Status message
            if overall_score >= 90:
                st.markdown("""
                <div class="success-box">
                    <h3>✅ Excellent! Project structure is highly compliant</h3>
                    <p>Your project follows Cookiecutter Data Science best practices and MLOps standards.</p>
                </div>
                """, unsafe_allow_html=True)
            elif overall_score >= 70:
                st.markdown("""
                <div class="warning-box">
                    <h3>⚠️ Good structure, minor improvements needed</h3>
                    <p>Most requirements are met, but some files/directories are missing.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="warning-box">
                    <h3>❌ Structure needs improvement</h3>
                    <p>Several required components are missing. Review the details below.</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Detailed metrics
            st.markdown("## 📈 Detailed Metrics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                dirs_pct = results['summary']['directories']['percentage']
                st.metric(
                    "Directories",
                    f"{results['summary']['directories']['present']}/{results['summary']['directories']['total']}",
                    f"{dirs_pct}%",
                    delta_color="normal" if dirs_pct >= 80 else "inverse"
                )
                st.progress(dirs_pct / 100)
            
            with col2:
                files_pct = results['summary']['files']['percentage']
                st.metric(
                    "Required Files",
                    f"{results['summary']['files']['present']}/{results['summary']['files']['total']}",
                    f"{files_pct}%",
                    delta_color="normal" if files_pct >= 80 else "inverse"
                )
                st.progress(files_pct / 100)
            
            with col3:
                mlops_pct = results['summary']['mlops']['percentage']
                st.metric(
                    "MLOps Files",
                    f"{results['summary']['mlops']['present']}/{results['summary']['mlops']['total']}",
                    f"{mlops_pct}%",
                    delta_color="normal" if mlops_pct >= 80 else "inverse"
                )
                st.progress(mlops_pct / 100)
            
            # Tabs for detailed breakdown
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📁 Directories", 
                "📄 Files", 
                "🔧 MLOps", 
                "🚫 .gitignore Status",
                "🌳 Tree View"
            ])
            
            with tab1:
                st.markdown("### Directory Structure")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### ✅ Present")
                    if results['directories']['present']:
                        for item in results['directories']['present']:
                            st.markdown(f"- ✅ `{item}`")
                    else:
                        st.info("No directories found")
                
                with col2:
                    st.markdown("#### ❌ Missing")
                    if results['directories']['missing']:
                        for item in results['directories']['missing']:
                            st.markdown(f"- ❌ `{item}`")
                    else:
                        st.success("All required directories present!")
            
            with tab2:
                st.markdown("### Required Files")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### ✅ Present")
                    if results['files']['present']:
                        for item in results['files']['present']:
                            st.markdown(f"- ✅ `{item}`")
                    else:
                        st.info("No files found")
                
                with col2:
                    st.markdown("#### ❌ Missing")
                    if results['files']['missing']:
                        for item in results['files']['missing']:
                            st.markdown(f"- ❌ `{item}`")
                    else:
                        st.success("All required files present!")
                
                # Optional files
                st.markdown("### 📋 Optional Files")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### ✅ Present")
                    if results['optional']['present']:
                        for item in results['optional']['present']:
                            st.markdown(f"- ✅ `{item}`")
                    else:
                        st.info("No optional files found")
                
                with col2:
                    st.markdown("#### ⚪ Not Present")
                    if results['optional']['missing']:
                        for item in results['optional']['missing']:
                            st.markdown(f"- ⚪ `{item}`")
            
            with tab3:
                st.markdown("### MLOps-Specific Files")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### ✅ Present")
                    if results['mlops']['present']:
                        for item in results['mlops']['present']:
                            st.markdown(f"- ✅ `{item}`")
                    else:
                        st.info("No MLOps files found")
                
                with col2:
                    st.markdown("#### ❌ Missing")
                    if results['mlops']['missing']:
                        for item in results['mlops']['missing']:
                            st.markdown(f"- ❌ `{item}`")
                    else:
                        st.success("All MLOps files present!")
            
            with tab4:
                st.markdown("### .gitignore Status (Fase 2 Cleanup)")
                
                if 'gitignore_status' in results:
                    gitignore = results['gitignore_status']
                    
                    st.markdown("""
                    <div class="info-box">
                        <p><strong>Best Practice:</strong> Artifacts like mlruns/, mlartifacts/, dvcstore/, 
                        and *.egg-info/ should NOT be versioned in Git.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### ✅ Properly Ignored")
                        if gitignore['properly_ignored']:
                            for item in gitignore['properly_ignored']:
                                st.markdown(f"- ✅ `{item}/` is gitignored")
                        else:
                            st.success("No artifacts need to be ignored (or already cleaned)")
                    
                    with col2:
                        st.markdown("#### ⚠️ Should Be Ignored")
                        if gitignore['should_be_ignored']:
                            for item in gitignore['should_be_ignored']:
                                st.markdown(f"- ⚠️ `{item}/` exists but not gitignored")
                        else:
                            st.success("All artifacts properly gitignored!")
                    
                    if gitignore.get('note'):
                        st.info(gitignore['note'])
            
            with tab5:
                st.markdown("### Repository Tree Structure")
                st.markdown("""
                <div class="info-box">
                    <p>Tree view shows the clean directory structure (hidden files and ignored dirs omitted).</p>
                </div>
                """, unsafe_allow_html=True)
                
                tree = validator.get_tree_structure(max_depth=3)
                st.code(tree, language="text")
            
            # Download results
            st.markdown("## 💾 Export Results")
            
            results_json = json.dumps(results, indent=2)
            st.download_button(
                label="📥 Download Results (JSON)",
                data=results_json,
                file_name=f"validation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
            
        except Exception as e:
            st.error(f"❌ Error during validation: {str(e)}")
            st.exception(e)

else:
    st.info("👈 Click 'Run Validation' in the sidebar to start")
    
    st.markdown("""
    ## 🎯 About This Dashboard
    
    This dashboard validates your MLOps project structure against **Cookiecutter Data Science** 
    best practices and MLOps standards.
    
    ### ✨ What's New in Fase 2
    
    - ✅ **Repository Cleanup**: Validates that artifacts are properly gitignored
    - ✅ **Professional Structure**: Recognizes `acoustic_ml/` module
    - ✅ **MLOps Standards**: Checks DVC, MLflow, and versioning setup
    - ✅ **Comprehensive Metrics**: Overall compliance score
    
    ### 📊 Validation Criteria
    
    The dashboard checks for:
    
    1. **Directory Structure** (data/, models/, notebooks/, etc.)
    2. **Required Files** (README.md, requirements.txt, etc.)
    3. **MLOps Files** (dvc.yaml, params.yaml, data.dvc)
    4. **.gitignore Status** (artifacts properly ignored)
    5. **Optional Files** (LICENSE, docker-compose.yml, etc.)
    
    ### 🏆 Scoring
    
    - **90-100%**: Excellent compliance ✅
    - **70-89%**: Good, minor improvements needed ⚠️
    - **<70%**: Structure needs improvement ❌
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p><strong>MLOps Team 24</strong> | Tecnológico de Monterrey | Fase 2 - 2024</p>
    <p>Developed with ❤️ using Streamlit</p>
</div>
""", unsafe_allow_html=True)
