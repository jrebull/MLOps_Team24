"""
MLOps Project Structure Validator - Professional Edition
Reconoce estructuras profesionales como acoustic_ml/ y pyproject.toml
Team 24 - Tecnológico de Monterrey
"""

import os
from pathlib import Path
from typing import Dict, List
import json


class CookieCutterValidator:
    """Validates project structure - recognizes professional alternatives"""
    
    EXPECTED_STRUCTURE = {
        'directories': [
            'data',
            'data/raw',
            'data/processed',
            'data/external',
            'notebooks',
            ('src', 'acoustic_ml', 'app'),  # Acepta cualquiera de estos
            'models',
            'reports',
            'reports/figures',
            'references',
            'docs',
        ],
        'files': [
            'README.md',
            'requirements.txt',
            ('.gitignore', '.dvcignore'),  # Acepta cualquiera
            ('Makefile', 'pyproject.toml'),  # Acepta cualquiera
            ('setup.py', 'pyproject.toml'),  # Acepta cualquiera
        ],
        'optional': [
            'LICENSE',
            'tox.ini',
            'setup.cfg',
            '.env',
            'data.dvc',
            '.dvc/.gitignore',
            '.dvc/config',
            'dvc.yaml',
            'params.yaml',
            'mlruns',
        ]
    }
    
    MLOPS_FILES = [
        'dvc.yaml',
        'params.yaml',
        '.dvc/config',
        'requirements.txt',
    ]
    
    def __init__(self, repo_path: str = '.'):
        self.repo_path = Path(repo_path).resolve()
        
    def validate(self) -> Dict:
        results = {
            'repo_path': str(self.repo_path),
            'exists': self.repo_path.exists(),
            'directories': self._check_directories(),
            'files': self._check_files(),
            'optional': self._check_optional(),
            'mlops': self._check_mlops(),
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
            }
        }
        
        total_items = total_dirs + total_files + total_mlops
        present_items = present_dirs + present_files + present_mlops
        results['summary']['overall'] = {
            'total': total_items,
            'present': present_items,
            'score': round((present_items / total_items) * 100, 2)
        }
        
        return results
    
    def _check_directories(self) -> Dict[str, List[str]]:
        present = []
        missing = []
        
        for item in self.EXPECTED_STRUCTURE['directories']:
            # Si es tupla, acepta cualquiera de las alternativas
            if isinstance(item, tuple):
                found = False
                for alt in item:
                    if (self.repo_path / alt).exists():
                        present.append(f"{item[0]} (found: {alt})")
                        found = True
                        break
                if not found:
                    missing.append(f"{item[0]} (or alternatives)")
            else:
                if (self.repo_path / item).exists():
                    present.append(item)
                else:
                    missing.append(item)
        
        return {'present': present, 'missing': missing}
    
    def _check_files(self) -> Dict[str, List[str]]:
        present = []
        missing = []
        
        for item in self.EXPECTED_STRUCTURE['files']:
            if isinstance(item, tuple):
                found = False
                for alt in item:
                    if (self.repo_path / alt).exists():
                        present.append(f"{item[0]} (found: {alt})")
                        found = True
                        break
                if not found:
                    missing.append(f"{item[0]} (or alternatives)")
            else:
                if (self.repo_path / item).exists():
                    present.append(item)
                else:
                    missing.append(item)
        
        return {'present': present, 'missing': missing}
    
    def _check_optional(self) -> Dict[str, List[str]]:
        present = []
        missing = []
        
        for item in self.EXPECTED_STRUCTURE['optional']:
            if (self.repo_path / item).exists():
                present.append(item)
            else:
                missing.append(item)
        
        return {'present': present, 'missing': missing}
    
    def _check_mlops(self) -> Dict[str, List[str]]:
        present = []
        missing = []
        
        for file in self.MLOPS_FILES:
            if (self.repo_path / file).exists():
                present.append(file)
            else:
                missing.append(file)
        
        return {'present': present, 'missing': missing}


    def get_tree_structure(self, max_depth: int = 3) -> str:
        """Generate tree view of repository"""
        def _tree(directory: Path, prefix: str = "", depth: int = 0) -> List[str]:
            if depth > max_depth:
                return []
            
            lines = []
            try:
                items = sorted(directory.iterdir(), key=lambda x: (not x.is_dir(), x.name))
                items = [item for item in items if not item.name.startswith('.')]
                
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


def validate_cookiecutter_structure(repo_path: str = '.') -> Dict:
    validator = CookieCutterValidator(repo_path)
    return validator.validate()


if __name__ == "__main__":
    import sys
    repo_path = sys.argv[1] if len(sys.argv) > 1 else '.'
    validator = CookieCutterValidator(repo_path)
    results = validator.validate()
    print(json.dumps(results, indent=2))