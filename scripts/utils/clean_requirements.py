#!/usr/bin/env python3
"""
Clean Requirements Generator
=============================
Analyzes all Python files in the project and generates a clean requirements.txt
with only the packages actually used in the code.

Usage:
    python scripts/utils/clean_requirements.py
    python scripts/utils/clean_requirements.py --dry-run
    python scripts/utils/clean_requirements.py --output requirements_clean.txt
"""

import ast
import re
import sys
from pathlib import Path
from typing import Set, Dict, List
import subprocess


# Mapping of import names to PyPI package names (when they differ)
IMPORT_TO_PACKAGE = {
    'sklearn': 'scikit-learn',
    'cv2': 'opencv-python',
    'PIL': 'Pillow',
    'yaml': 'PyYAML',
    'dotenv': 'python-dotenv',
    'dateutil': 'python-dateutil',
    'googleapiclient': 'google-api-python-client',
    'bs4': 'beautifulsoup4',
    'tensorflow_macos': 'tensorflow-macos',
    'tensorflow_metal': 'tensorflow-metal',
}

class RequirementsAnalyzer:
    """Analyze Python imports and generate clean requirements"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.imports_found = set()
        self.current_requirements = {}
        self.standard_lib = self._get_standard_lib()
        
    def _get_standard_lib(self) -> Set[str]:
        """Get Python standard library modules"""
        # Python 3.12 standard library (most common ones)
        return {
            'abc', 'aifc', 'argparse', 'array', 'ast', 'asynchat', 'asyncio',
            'asyncore', 'atexit', 'audioop', 'base64', 'bdb', 'binascii',
            'binhex', 'bisect', 'builtins', 'bz2', 'calendar', 'cgi', 'cgitb',
            'chunk', 'cmath', 'cmd', 'code', 'codecs', 'codeop', 'collections',
            'colorsys', 'compileall', 'concurrent', 'configparser', 'contextlib',
            'contextvars', 'copy', 'copyreg', 'cProfile', 'crypt', 'csv', 'ctypes',
            'curses', 'dataclasses', 'datetime', 'dbm', 'decimal', 'difflib',
            'dis', 'distutils', 'doctest', 'email', 'encodings', 'enum', 'errno',
            'faulthandler', 'fcntl', 'filecmp', 'fileinput', 'fnmatch', 'formatter',
            'fractions', 'ftplib', 'functools', 'gc', 'getopt', 'getpass', 'gettext',
            'glob', 'graphlib', 'grp', 'gzip', 'hashlib', 'heapq', 'hmac', 'html',
            'http', 'imaplib', 'imghdr', 'imp', 'importlib', 'inspect', 'io',
            'ipaddress', 'itertools', 'json', 'keyword', 'lib2to3', 'linecache',
            'locale', 'logging', 'lzma', 'mailbox', 'mailcap', 'marshal', 'math',
            'mimetypes', 'mmap', 'modulefinder', 'msilib', 'msvcrt', 'multiprocessing',
            'netrc', 'nis', 'nntplib', 'numbers', 'operator', 'optparse', 'os',
            'ossaudiodev', 'parser', 'pathlib', 'pdb', 'pickle', 'pickletools',
            'pipes', 'pkgutil', 'platform', 'plistlib', 'poplib', 'posix',
            'posixpath', 'pprint', 'profile', 'pstats', 'pty', 'pwd', 'py_compile',
            'pyclbr', 'pydoc', 'queue', 'quopri', 'random', 're', 'readline',
            'reprlib', 'resource', 'rlcompleter', 'runpy', 'sched', 'secrets',
            'select', 'selectors', 'shelve', 'shlex', 'shutil', 'signal', 'site',
            'smtpd', 'smtplib', 'sndhdr', 'socket', 'socketserver', 'spwd', 'sqlite3',
            'ssl', 'stat', 'statistics', 'string', 'stringprep', 'struct', 'subprocess',
            'sunau', 'symbol', 'symtable', 'sys', 'sysconfig', 'syslog', 'tabnanny',
            'tarfile', 'telnetlib', 'tempfile', 'termios', 'test', 'textwrap',
            'threading', 'time', 'timeit', 'token', 'tokenize', 'tomllib', 'trace',
            'traceback', 'tracemalloc', 'tty', 'turtle', 'turtledemo', 'types',
            'typing', 'typing_extensions', 'unicodedata', 'unittest', 'urllib',
            'uu', 'uuid', 'venv', 'warnings', 'wave', 'weakref', 'webbrowser',
            'winreg', 'winsound', 'wsgiref', 'xdrlib', 'xml', 'xmlrpc', 'zipapp',
            'zipfile', 'zipimport', 'zlib', '_thread',
        }
    
    def find_python_files(self) -> List[Path]:
        """Find all Python files in the project"""
        print("🔍 Scanning for Python files...")
        
        # Directories to exclude (more comprehensive)
        exclude_dirs = {
            '.git', '.venv', 'venv', 'env', '__pycache__', 
            '.pytest_cache', '.mypy_cache', 'node_modules',
            'build', 'dist', '.egg-info', '.dvc', 'mlruns',
            'site-packages', 'lib', 'lib64', 'include', 'bin',
            '.tox', '.eggs', 'htmlcov', '.coverage'
        }
        
        python_files = []
        
        # Get all directories at root level to check
        root_items = list(self.project_root.iterdir())
        
        for item in root_items:
            # Skip if it's an excluded directory at root level
            if item.is_dir() and item.name in exclude_dirs:
                continue
            
            # If it's a directory, search recursively
            if item.is_dir():
                for py_file in item.rglob("*.py"):
                    # Double-check: skip if any part of path matches excluded dirs
                    relative_parts = py_file.relative_to(self.project_root).parts
                    if any(part in exclude_dirs for part in relative_parts):
                        continue
                    python_files.append(py_file)
            
            # If it's a Python file at root level
            elif item.suffix == '.py':
                python_files.append(item)
        
        print(f"   Found {len(python_files)} Python files")
        return python_files
    
    def extract_imports_from_file(self, file_path: Path) -> Set[str]:
        """Extract import statements from a Python file"""
        imports = set()
        
        try:
            # Try UTF-8 first, then fallback to other encodings
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except UnicodeDecodeError:
                # Try with latin-1 as fallback
                try:
                    with open(file_path, 'r', encoding='latin-1') as f:
                        content = f.read()
                except:
                    # Skip files we can't decode
                    return imports
            
            # Parse the AST
            tree = ast.parse(content, filename=str(file_path))
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        # Get top-level package name
                        package = alias.name.split('.')[0]
                        imports.add(package)
                        
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        # Get top-level package name
                        package = node.module.split('.')[0]
                        imports.add(package)
        
        except SyntaxError:
            # Skip files with syntax errors (like Python 2 code)
            pass
        except Exception:
            # Skip any other problematic files silently
            pass
        
        return imports
    
    def analyze_all_files(self) -> Set[str]:
        """Analyze all Python files and collect imports"""
        print("\n📦 Analyzing imports...")
        
        python_files = self.find_python_files()
        all_imports = set()
        
        for py_file in python_files:
            file_imports = self.extract_imports_from_file(py_file)
            all_imports.update(file_imports)
        
        # Filter out standard library and local imports
        external_imports = {
            imp for imp in all_imports 
            if imp not in self.standard_lib 
            and not imp.startswith('acoustic_ml')
            and not imp.startswith('test')
            and imp != 'setup'
        }
        
        print(f"   Found {len(external_imports)} external package imports")
        return external_imports
    
    def load_current_requirements(self):
        """Load current requirements.txt"""
        print("\n📋 Reading current requirements.txt...")
        
        req_file = self.project_root / 'requirements.txt'
        if not req_file.exists():
            print("   ⚠️  requirements.txt not found")
            return
        
        with open(req_file, 'r') as f:
            for line in f:
                line = line.strip()
                
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                
                # Parse package name and version
                # Handle ==, >=, <=, ~=, etc.
                match = re.match(r'^([a-zA-Z0-9\-_\.]+)', line)
                if match:
                    package_name = match.group(1)
                    self.current_requirements[package_name.lower()] = line
        
        print(f"   Found {len(self.current_requirements)} packages in requirements.txt")
    
    def map_import_to_package(self, import_name: str) -> str:
        """Map import name to PyPI package name"""
        return IMPORT_TO_PACKAGE.get(import_name, import_name)
    
    def generate_clean_requirements(self, imports: Set[str]) -> List[str]:
        """Generate clean requirements from imports"""
        print("\n✨ Generating clean requirements...")
        
        clean_reqs = []
        found_packages = []
        missing_packages = []
        
        for import_name in sorted(imports):
            package_name = self.map_import_to_package(import_name)
            
            # Try to find in current requirements (case-insensitive)
            found = False
            for req_name, req_line in self.current_requirements.items():
                if req_name == package_name.lower():
                    clean_reqs.append(req_line)
                    found_packages.append(package_name)
                    found = True
                    break
            
            if not found:
                missing_packages.append(package_name)
        
        print(f"   ✅ Matched {len(found_packages)} packages")
        if missing_packages:
            print(f"   ⚠️  Missing in requirements.txt: {', '.join(missing_packages)}")
            print(f"      (These are imported but not in requirements.txt)")
        
        return clean_reqs
    
    def save_requirements(self, requirements: List[str], output_file: str = "requirements_clean.txt"):
        """Save clean requirements to file"""
        output_path = self.project_root / output_file
        
        with open(output_path, 'w') as f:
            f.write("# Auto-generated clean requirements\n")
            f.write("# Generated by clean_requirements.py\n")
            f.write(f"# Total packages: {len(requirements)}\n\n")
            
            # Group by category (basic heuristic)
            ml_packages = []
            data_packages = []
            dev_packages = []
            api_packages = []
            other_packages = []
            
            for req in requirements:
                req_lower = req.lower()
                if any(x in req_lower for x in ['mlflow', 'dvc', 'sklearn', 'tensorflow', 'torch', 'keras']):
                    ml_packages.append(req)
                elif any(x in req_lower for x in ['pandas', 'numpy', 'scipy', 'matplotlib', 'seaborn']):
                    data_packages.append(req)
                elif any(x in req_lower for x in ['pytest', 'jupyter', 'ipython', 'notebook']):
                    dev_packages.append(req)
                elif any(x in req_lower for x in ['fastapi', 'uvicorn', 'pydantic', 'httpx']):
                    api_packages.append(req)
                else:
                    other_packages.append(req)
            
            if data_packages:
                f.write("# Data Science Core\n")
                f.write("\n".join(data_packages) + "\n\n")
            
            if ml_packages:
                f.write("# MLOps & Machine Learning\n")
                f.write("\n".join(ml_packages) + "\n\n")
            
            if api_packages:
                f.write("# API & Web\n")
                f.write("\n".join(api_packages) + "\n\n")
            
            if dev_packages:
                f.write("# Development & Testing\n")
                f.write("\n".join(dev_packages) + "\n\n")
            
            if other_packages:
                f.write("# Other Dependencies\n")
                f.write("\n".join(other_packages) + "\n")
        
        print(f"\n💾 Saved to: {output_path}")
    
    def run(self, output_file: str = "requirements_clean.txt", dry_run: bool = False):
        """Run the full analysis"""
        print("=" * 60)
        print("🧹 Clean Requirements Generator")
        print("=" * 60)
        
        # Step 1: Find all imports
        imports = self.analyze_all_files()
        
        # Step 2: Load current requirements
        self.load_current_requirements()
        
        # Step 3: Generate clean requirements
        clean_reqs = self.generate_clean_requirements(imports)
        
        # Step 4: Show statistics
        print("\n📊 Statistics:")
        print(f"   Total imports found: {len(imports)}")
        print(f"   Clean requirements: {len(clean_reqs)}")
        print(f"   Original requirements: {len(self.current_requirements)}")
        print(f"   Removed packages: {len(self.current_requirements) - len(clean_reqs)}")
        
        # Step 5: Save (if not dry run)
        if dry_run:
            print("\n🔍 DRY RUN - Clean requirements:")
            for req in clean_reqs:
                print(f"   {req}")
        else:
            self.save_requirements(clean_reqs, output_file)
            
            print("\n✅ Done!")
            print(f"\n💡 Next steps:")
            print(f"   1. Review: cat {output_file}")
            print(f"   2. Backup: cp requirements.txt requirements_backup.txt")
            print(f"   3. Replace: mv {output_file} requirements.txt")
            print(f"   4. Reinstall: pip install -r requirements.txt")


def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate clean requirements.txt from actual imports"
    )
    parser.add_argument(
        '--output',
        default='requirements_clean.txt',
        help='Output file name (default: requirements_clean.txt)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be generated without saving'
    )
    
    args = parser.parse_args()
    
    # Get project root (current directory)
    project_root = Path.cwd()
    
    # Run analyzer
    analyzer = RequirementsAnalyzer(project_root)
    analyzer.run(output_file=args.output, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
