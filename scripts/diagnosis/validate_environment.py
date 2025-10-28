#!/usr/bin/env python3
"""
MLOps Environment Validation Script
====================================
Professional diagnostic tool for MLOps Team24 project.

Usage:
    python scripts/diagnosis/validate_environment.py
    python scripts/diagnosis/validate_environment.py --json
    python scripts/diagnosis/validate_environment.py --save-report
"""

import sys
import os
import subprocess
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import importlib.util


class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class MLOpsValidator:
    """MLOps environment validation and diagnostics"""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.results = {}
        self.issues = []
        self.project_root = Path.cwd()
        
    def print_section(self, title: str, emoji: str = "📋"):
        """Print formatted section header"""
        if self.verbose:
            print(f"\n{Colors.BOLD}{emoji} {title}{Colors.ENDC}")
            print("=" * 60)
    
    def print_success(self, message: str):
        """Print success message"""
        if self.verbose:
            print(f"{Colors.OKGREEN}✅ {message}{Colors.ENDC}")
    
    def print_warning(self, message: str):
        """Print warning message"""
        if self.verbose:
            print(f"{Colors.WARNING}⚠️  {message}{Colors.ENDC}")
        self.issues.append(message)
    
    def print_error(self, message: str):
        """Print error message"""
        if self.verbose:
            print(f"{Colors.FAIL}❌ {message}{Colors.ENDC}")
        self.issues.append(message)
    
    def print_info(self, message: str):
        """Print info message"""
        if self.verbose:
            print(f"{Colors.OKCYAN}ℹ️  {message}{Colors.ENDC}")
    
    def run_command(self, cmd: List[str], capture: bool = True) -> Tuple[int, str, str]:
        """Run shell command and return output"""
        try:
            result = subprocess.run(
                cmd,
                capture_output=capture,
                text=True,
                timeout=10
            )
            return result.returncode, result.stdout.strip(), result.stderr.strip()
        except Exception as e:
            return 1, "", str(e)
    
    def check_python_environment(self) -> Dict:
        """Validate Python and virtual environment"""
        self.print_section("Python Environment", "🐍")
        
        env_info = {}
        
        # Check Python version
        python_version = sys.version.split()[0]
        env_info['python_version'] = python_version
        self.print_info(f"Python version: {python_version}")
        
        # Check virtual environment
        venv = os.environ.get('VIRTUAL_ENV')
        if venv:
            self.print_success(f"Virtual environment active: {venv}")
            env_info['venv_active'] = True
            env_info['venv_path'] = venv
        else:
            self.print_warning("Virtual environment NOT active")
            env_info['venv_active'] = False
        
        # Check Python executable
        python_path = sys.executable
        env_info['python_path'] = python_path
        self.print_info(f"Python executable: {python_path}")
        
        self.results['python_environment'] = env_info
        return env_info
    
    def check_dependencies(self) -> Dict:
        """Check Python package dependencies"""
        self.print_section("Python Dependencies", "📦")
        
        dependencies = {}
        # Core dependencies confirmed to be in active use
        # Based on actual code analysis and grep verification
        required_packages = [
            # Data Science Core
            'pandas',
            'numpy',
            'scipy',
            'sklearn',  # scikit-learn
            'joblib',
            
            # Visualization
            'matplotlib',
            'seaborn',
            
            # MLOps Infrastructure
            'mlflow',
            'dvc',
            
            # API & Dashboard
            'fastapi',
            'pydantic',
            'streamlit',
            
            # Testing
            'pytest'
        ]
        
        for package in required_packages:
            try:
                # Try to import the package
                if package == 'sklearn':
                    import sklearn
                    version = sklearn.__version__
                else:
                    mod = importlib.import_module(package)
                    version = getattr(mod, '__version__', 'unknown')
                
                self.print_success(f"{package} {version}")
                dependencies[package] = {
                    'installed': True,
                    'version': version
                }
            except ImportError:
                self.print_error(f"{package} NOT installed")
                dependencies[package] = {
                    'installed': False,
                    'version': None
                }
        
        self.results['dependencies'] = dependencies
        return dependencies
    
    def check_acoustic_ml(self) -> Dict:
        """Check acoustic_ml custom module"""
        self.print_section("Acoustic ML Module", "🎵")
        
        module_info = {}
        
        try:
            import acoustic_ml
            version = getattr(acoustic_ml, '__version__', 'unknown')
            module_path = Path(acoustic_ml.__file__).parent
            
            self.print_success(f"acoustic_ml v{version}")
            self.print_info(f"Location: {module_path}")
            
            module_info['installed'] = True
            module_info['version'] = version
            module_info['path'] = str(module_path)
            
        except ImportError as e:
            self.print_error("acoustic_ml NOT working")
            self.print_info("Run: pip install -e .")
            module_info['installed'] = False
            module_info['error'] = str(e)
        
        self.results['acoustic_ml'] = module_info
        return module_info
    
    def check_git_status(self) -> Dict:
        """Check Git repository status"""
        self.print_section("Git Repository", "📂")
        
        git_info = {}
        
        # Check current branch
        ret, branch, _ = self.run_command(['git', 'branch', '--show-current'])
        if ret == 0:
            self.print_info(f"Current branch: {branch}")
            git_info['branch'] = branch
        
        # Check last commit
        ret, commit, _ = self.run_command(['git', 'log', '-1', '--oneline'])
        if ret == 0:
            self.print_info(f"Last commit: {commit}")
            git_info['last_commit'] = commit
        
        # Check status
        ret, status, _ = self.run_command(['git', 'status', '--short'])
        if ret == 0:
            if status:
                self.print_warning(f"Uncommitted changes: {len(status.splitlines())} files")
                git_info['clean'] = False
                git_info['uncommitted_files'] = len(status.splitlines())
            else:
                self.print_success("Working tree clean")
                git_info['clean'] = True
        
        self.results['git'] = git_info
        return git_info
    
    def check_dvc_status(self) -> Dict:
        """Check DVC data versioning status"""
        self.print_section("DVC Data Versioning", "💾")
        
        dvc_info = {}
        
        # Check if DVC is installed
        ret, version, _ = self.run_command(['dvc', 'version'])
        if ret == 0:
            self.print_success(f"DVC installed: {version.split()[0]}")
            dvc_info['installed'] = True
            dvc_info['version'] = version.split()[0]
        else:
            self.print_error("DVC not installed")
            dvc_info['installed'] = False
            self.results['dvc'] = dvc_info
            return dvc_info
        
        # Check DVC status
        ret, status, _ = self.run_command(['dvc', 'status'])
        if ret == 0:
            if "up to date" in status.lower() or not status:
                self.print_success("Data and pipelines up to date")
                dvc_info['synced'] = True
            else:
                self.print_warning("DVC data not synced")
                dvc_info['synced'] = False
        
        # Check remote
        ret, remotes, _ = self.run_command(['dvc', 'remote', 'list'])
        if ret == 0 and remotes:
            self.print_success(f"DVC remote configured: {remotes.split()[0]}")
            dvc_info['remote_configured'] = True
            dvc_info['remote_name'] = remotes.split()[0]
        else:
            self.print_warning("No DVC remote configured")
            dvc_info['remote_configured'] = False
        
        # Count .dvc files
        dvc_files = list(self.project_root.rglob("*.dvc"))
        dvc_files = [f for f in dvc_files if '.git' not in str(f)]
        self.print_info(f"DVC tracked files: {len(dvc_files)}")
        dvc_info['tracked_files'] = len(dvc_files)
        
        self.results['dvc'] = dvc_info
        return dvc_info
    
    def check_project_structure(self) -> Dict:
        """Check project directory structure"""
        self.print_section("Project Structure", "📁")
        
        structure = {}
        required_dirs = [
            'data',
            'data/raw',
            'data/processed',
            'models',
            'acoustic_ml',  # Custom module instead of src/
            'tests',
            'mlruns',
            'scripts'
        ]
        
        for dir_path in required_dirs:
            path = self.project_root / dir_path
            exists = path.exists()
            structure[dir_path] = exists
            
            if exists:
                self.print_success(f"{dir_path}/ exists")
            else:
                self.print_warning(f"{dir_path}/ NOT found")
        
        self.results['project_structure'] = structure
        return structure
    
    def check_mlflow(self) -> Dict:
        """Check MLflow experiment tracking"""
        self.print_section("MLflow Experiments", "📊")
        
        mlflow_info = {}
        mlruns_path = self.project_root / 'mlruns'
        
        if mlruns_path.exists():
            # Count experiments
            experiments = [d for d in mlruns_path.iterdir() 
                          if d.is_dir() and d.name not in ['.trash', '0']]
            
            self.print_success(f"mlruns/ exists ({len(experiments)} experiments)")
            mlflow_info['exists'] = True
            mlflow_info['experiment_count'] = len(experiments)
            
            # Find latest run
            runs = list(mlruns_path.rglob("*/*/meta.yaml"))
            if runs:
                latest = max(runs, key=lambda p: p.stat().st_mtime)
                self.print_info(f"Latest run: {latest.parent.parent.name}")
                mlflow_info['has_runs'] = True
        else:
            self.print_warning("mlruns/ NOT found")
            mlflow_info['exists'] = False
        
        self.results['mlflow'] = mlflow_info
        return mlflow_info
    
    def check_important_files(self) -> Dict:
        """Check for important project files"""
        self.print_section("Important Files", "📄")
        
        files_status = {}
        important_files = [
            'requirements.txt',
            'pyproject.toml',  # Modern Python packaging (PEP 518)
            'dvc.yaml',
            '.dvc/config',
            'README.md',
            '.gitignore'
        ]
        
        for file_path in important_files:
            path = self.project_root / file_path
            exists = path.exists()
            files_status[file_path] = exists
            
            if exists:
                self.print_success(f"{file_path} exists")
            else:
                self.print_warning(f"{file_path} NOT found")
        
        self.results['important_files'] = files_status
        return files_status
    
    def generate_summary(self) -> Dict:
        """Generate final summary"""
        self.print_section("Summary", "✨")
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'project_root': str(self.project_root),
            'total_issues': len(self.issues),
            'issues': self.issues
        }
        
        if len(self.issues) == 0:
            self.print_success("🎉 ALL CHECKS PASSED!")
            self.print_info("Environment is ready for MLOps work")
            summary['status'] = 'PASSED'
        else:
            self.print_warning(f"Found {len(self.issues)} issue(s)")
            self.print_info("Review details above")
            summary['status'] = 'ISSUES_FOUND'
            
            if self.verbose:
                print("\n" + Colors.WARNING + "Issues found:" + Colors.ENDC)
                for i, issue in enumerate(self.issues, 1):
                    print(f"  {i}. {issue}")
        
        self.results['summary'] = summary
        return summary
    
    def run_all_checks(self) -> Dict:
        """Run all validation checks"""
        print(f"\n{Colors.BOLD}{Colors.HEADER}")
        print("=" * 60)
        print("🔍 MLOps Team24 Environment Validation")
        print("=" * 60)
        print(Colors.ENDC)
        
        self.check_python_environment()
        self.check_dependencies()
        self.check_acoustic_ml()
        self.check_git_status()
        self.check_dvc_status()
        self.check_project_structure()
        self.check_mlflow()
        self.check_important_files()
        self.generate_summary()
        
        print(f"\n{Colors.BOLD}{'=' * 60}{Colors.ENDC}\n")
        
        return self.results
    
    def save_report(self, filename: Optional[str] = None):
        """Save validation report to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"validation_report_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        self.print_success(f"Report saved to: {filename}")


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="MLOps Environment Validation Tool"
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output results in JSON format'
    )
    parser.add_argument(
        '--save-report',
        action='store_true',
        help='Save validation report to file'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )
    
    args = parser.parse_args()
    
    validator = MLOpsValidator(verbose=not args.quiet)
    results = validator.run_all_checks()
    
    if args.json:
        print(json.dumps(results, indent=2))
    
    if args.save_report:
        validator.save_report()
    
    # Exit with error code if issues found
    sys.exit(len(validator.issues))


if __name__ == "__main__":
    main()
