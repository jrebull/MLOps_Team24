"""
MLOps Project Structure Dashboard
Streamlit-based validation dashboard for Turkish Music Emotion Recognition project
Actualizado para Fase 2 - Post Cleanup
Team 24 - Tecnológico de Monterrey (ITESM)

All-in-one version: validator + dashboard in single file
"""
import streamlit as st
import os
from pathlib import Path
import json
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px


# ============================================================================
# VALIDATOR CLASS (embedded)
# ============================================================================

class CookieCutterValidator:
    """Validates project structure - recognizes professional alternatives"""
    
    EXPECTED_STRUCTURE = {
        'directories': [
            'data',
            'data/raw',
            'data/processed',
            'data/external',
            'data/interim',
            'notebooks',
            ('src', 'acoustic_ml'),  # Acepta src/ o acoustic_ml/
            'models',
            'reports',
            'reports/figures',
            'references',
            'docs',
            'scripts',
            'tests',
            'app',  # FastAPI app
            'monitoring',  # Monitoring & dashboards
        ],
        'files': [
            'README.md',
            'requirements.txt',
            ('.gitignore', '.dvcignore'),
            ('Makefile', 'pyproject.toml'),
            ('setup.py', 'pyproject.toml'),
        ],
        'optional': [
            'LICENSE',
            'tox.ini',
            'setup.cfg',
            '.pre-commit-config.yaml',
            'docker-compose.yml',
            'Dockerfile',
            '.env.example',
            'config.env',
        ],
    }
    
    MLOPS_FILES = [
        'dvc.yaml',
        'params.yaml',
        '.dvc/config',
        'requirements.txt',
        'data.dvc',
    ]
    
    SHOULD_BE_GITIGNORED = [
        'mlruns',
        'mlartifacts', 
        'dvcstore',
        'acoustic_ml.egg-info',
        '__pycache__',
        '.pytest_cache',
        '.ipynb_checkpoints',
    ]

    def __init__(self, repo_path: str = '.'):
        # Navigate to repo root from dashboard directory
        current_path = Path(repo_path).resolve()
        
        # If we're in monitoring/dashboard, go up two levels
        if current_path.name == 'dashboard':
            current_path = current_path.parent.parent
        elif current_path.name == 'monitoring':
            current_path = current_path.parent
            
        self.repo_path = current_path
        
    def validate(self):
        results = {
            'repo_path': str(self.repo_path),
            'exists': self.repo_path.exists(),
            'directories': self._check_directories(),
            'files': self._check_files(),
            'optional': self._check_optional(),
            'mlops': self._check_mlops(),
            'gitignore_status': self._check_gitignore(),
            'summary': {}
        }
        
        total_dirs = len(self.EXPECTED_STRUCTURE['directories'])
        total_files = len(self.EXPECTED_STRUCTURE['files'])
        total_mlops = len(self.MLOPS_FILES)
        
        present_dirs = len(results['directories']['present'])
        present_files = len(results['files']['present'])
        present_mlops = len(results['mlops']['present'])
        
        results['summary'] = {
            'directories': {
                'total': total_dirs,
                'present': present_dirs,
                'missing': total_dirs - present_dirs,
                'percentage': round((present_dirs / total_dirs) * 100, 2)
            },
            'files': {
                'total': total_files,
                'present': present_files,
                'missing': total_files - present_files,
                'percentage': round((present_files / total_files) * 100, 2)
            },
            'mlops': {
                'total': total_mlops,
                'present': present_mlops,
                'missing': total_mlops - present_mlops,
                'percentage': round((present_mlops / total_mlops) * 100, 2)
            },
            'overall_score': round(
                (present_dirs / total_dirs + present_files / total_files + present_mlops / total_mlops) / 3 * 100, 2
            )
        }
        
        return results
    
    def _check_directories(self):
        present = []
        missing = []
        
        for item in self.EXPECTED_STRUCTURE['directories']:
            if isinstance(item, tuple):
                found = False
                for alt in item:
                    if (self.repo_path / alt).is_dir():
                        present.append(alt)
                        found = True
                        break
                if not found:
                    missing.append(f"{item[0]} (or alternatives)")
            else:
                if (self.repo_path / item).is_dir():
                    present.append(item)
                else:
                    missing.append(item)
        
        return {'present': present, 'missing': missing}
    
    def _check_files(self):
        present = []
        missing = []
        
        for item in self.EXPECTED_STRUCTURE['files']:
            if isinstance(item, tuple):
                found = False
                for alt in item:
                    if (self.repo_path / alt).is_file():
                        present.append(alt)
                        found = True
                        break
                if not found:
                    missing.append(f"{item[0]} (or alternatives)")
            else:
                if (self.repo_path / item).is_file():
                    present.append(item)
                else:
                    missing.append(item)
        
        return {'present': present, 'missing': missing}
    
    def _check_optional(self):
        present = []
        missing = []
        
        for file in self.EXPECTED_STRUCTURE['optional']:
            if (self.repo_path / file).exists():
                present.append(file)
            else:
                missing.append(file)
        
        return {'present': present, 'missing': missing}
    
    def _check_mlops(self):
        present = []
        missing = []
        
        for file in self.MLOPS_FILES:
            if (self.repo_path / file).exists():
                present.append(file)
            else:
                missing.append(file)
        
        return {'present': present, 'missing': missing}
    
    def _check_gitignore(self):
        properly_ignored = []
        should_be_ignored = []
        
        gitignore_path = self.repo_path / '.gitignore'
        
        if not gitignore_path.exists():
            return {
                'properly_ignored': [],
                'should_be_ignored': self.SHOULD_BE_GITIGNORED,
                'note': '.gitignore file not found'
            }
        
        gitignore_content = gitignore_path.read_text()
        
        for item in self.SHOULD_BE_GITIGNORED:
            item_path = self.repo_path / item
            if item_path.exists():
                if item in gitignore_content or f"{item}/" in gitignore_content:
                    properly_ignored.append(item)
                else:
                    should_be_ignored.append(item)
        
        return {
            'properly_ignored': properly_ignored,
            'should_be_ignored': should_be_ignored,
            'note': f'Checked {len(self.SHOULD_BE_GITIGNORED)} patterns'
        }
    
    def get_tree_structure(self, max_depth: int = 3):
        def _tree(directory: Path, prefix: str = "", depth: int = 0):
            if depth > max_depth:
                return []
            
            lines = []
            try:
                items = sorted(directory.iterdir(), key=lambda x: (not x.is_dir(), x.name))
                items = [item for item in items if not item.name.startswith('.')]
                items = [item for item in items if item.name not in ['__pycache__', 'mlruns', 'mlartifacts', 'dvcstore']]
                
                for i, item in enumerate(items):
                    is_last = i == len(items) - 1
                    current_prefix = "└── " if is_last else "├── "
                    lines.append(f"{prefix}{current_prefix}{item.name}")
                    
                    if item.is_dir():
                        extension = "    " if is_last else "│   "
                        lines.extend(_tree(item, prefix + extension, depth + 1))
            except PermissionError:
                pass
            
            return lines
        
        tree_lines = [f"{self.repo_path.name}/"]
        tree_lines.extend(_tree(self.repo_path))
        return "\n".join(tree_lines)


# ============================================================================
# STREAMLIT DASHBOARD
# ============================================================================

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
    st.markdown("### 📊 Dashboard Options")
    
    # Try to detect repo root
    current_file = Path(__file__).resolve()
    if current_file.parent.name == 'dashboard':
        default_path = str(current_file.parent.parent.parent)
    else:
        default_path = "."
    
    repo_path = st.text_input("Repository Path", value=default_path, help="Root path of the repository")
    
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
