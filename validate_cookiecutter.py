#!/usr/bin/env python3
"""
Script de Validación Cookiecutter Data Science
MLOps Team 24 - Turkish Music Emotion Recognition
Fase 2 - Avance de Proyecto

Uso:
    python validate_cookiecutter.py

Este script valida que tu proyecto esté 100% alineado con Cookiecutter Data Science.
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple, Dict
import subprocess
import json

# ANSI color codes
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
BOLD = '\033[1m'
RESET = '\033[0m'


class CookiecutterValidator:
    """Valida la estructura Cookiecutter Data Science de un proyecto."""
    
    def __init__(self, project_root: Path = None):
        """Inicializa el validador con la raíz del proyecto."""
        self.project_root = project_root or Path.cwd()
        self.results = {
            "total_checks": 0,
            "passed": 0,
            "failed": 0,
            "warnings": 0,
            "details": []
        }
    
    def check_directory(self, path: str, description: str, required: bool = True) -> bool:
        """Verifica si un directorio existe."""
        self.results["total_checks"] += 1
        dir_path = self.project_root / path
        exists = dir_path.exists() and dir_path.is_dir()
        
        if exists:
            self.results["passed"] += 1
            self._log_success(f"✓ {description}: {path}")
            return True
        elif required:
            self.results["failed"] += 1
            self._log_error(f"✗ {description}: {path} NO ENCONTRADO")
            return False
        else:
            self.results["warnings"] += 1
            self._log_warning(f"⚠ {description}: {path} no encontrado (opcional)")
            return False
    
    def check_file(self, path: str, description: str, required: bool = True) -> bool:
        """Verifica si un archivo existe."""
        self.results["total_checks"] += 1
        file_path = self.project_root / path
        exists = file_path.exists() and file_path.is_file()
        
        if exists:
            self.results["passed"] += 1
            self._log_success(f"✓ {description}: {path}")
            return True
        elif required:
            self.results["failed"] += 1
            self._log_error(f"✗ {description}: {path} NO ENCONTRADO")
            return False
        else:
            self.results["warnings"] += 1
            self._log_warning(f"⚠ {description}: {path} no encontrado (opcional)")
            return False
    
    def check_git_repo(self) -> bool:
        """Verifica que sea un repositorio Git."""
        self.results["total_checks"] += 1
        git_dir = self.project_root / ".git"
        
        if git_dir.exists():
            self.results["passed"] += 1
            self._log_success("✓ Repositorio Git inicializado")
            return True
        else:
            self.results["failed"] += 1
            self._log_error("✗ NO es un repositorio Git")
            return False
    
    def check_dvc_config(self) -> bool:
        """Verifica la configuración de DVC."""
        self.results["total_checks"] += 1
        dvc_config = self.project_root / ".dvc" / "config"
        
        if dvc_config.exists():
            self.results["passed"] += 1
            self._log_success("✓ DVC configurado")
            return True
        else:
            self.results["warnings"] += 1
            self._log_warning("⚠ DVC no configurado (recomendado)")
            return False
    
    def check_python_module(self, module_name: str) -> bool:
        """Verifica que el módulo Python tenga la estructura correcta."""
        self.results["total_checks"] += 1
        module_dir = self.project_root / module_name
        
        if not module_dir.exists():
            self.results["failed"] += 1
            self._log_error(f"✗ Módulo Python '{module_name}/' NO ENCONTRADO")
            return False
        
        required_files = [
            "__init__.py",
            "config.py",
            "dataset.py",
            "features.py",
        ]
        
        all_exist = True
        for file in required_files:
            file_path = module_dir / file
            if not file_path.exists():
                self._log_error(f"  ✗ {module_name}/{file} faltante")
                all_exist = False
        
        # Check modeling subdirectory
        modeling_dir = module_dir / "modeling"
        if modeling_dir.exists():
            for file in ["__init__.py", "train.py", "predict.py"]:
                file_path = modeling_dir / file
                if not file_path.exists():
                    self._log_error(f"  ✗ {module_name}/modeling/{file} faltante")
                    all_exist = False
        else:
            self._log_error(f"  ✗ {module_name}/modeling/ NO ENCONTRADO")
            all_exist = False
        
        if all_exist:
            self.results["passed"] += 1
            self._log_success(f"✓ Módulo Python '{module_name}/' correctamente estructurado")
            return True
        else:
            self.results["failed"] += 1
            return False
    
    def check_notebooks_naming(self) -> bool:
        """Verifica que los notebooks sigan la convención de nombres."""
        self.results["total_checks"] += 1
        notebooks_dir = self.project_root / "notebooks"
        
        if not notebooks_dir.exists():
            self.results["warnings"] += 1
            self._log_warning("⚠ No se encontró directorio notebooks/")
            return False
        
        notebooks = list(notebooks_dir.glob("*.ipynb"))
        if not notebooks:
            self.results["warnings"] += 1
            self._log_warning("⚠ No se encontraron notebooks en notebooks/")
            return False
        
        # Convention: #.#-initials-description.ipynb
        import re
        pattern = re.compile(r'^\d+\.\d+-[a-z]+-[\w-]+\.ipynb$')
        
        correct = []
        incorrect = []
        
        for nb in notebooks:
            if pattern.match(nb.name):
                correct.append(nb.name)
            else:
                incorrect.append(nb.name)
        
        if incorrect:
            self.results["warnings"] += 1
            self._log_warning(f"⚠ Notebooks con convención incorrecta:")
            for name in incorrect:
                print(f"    - {name}")
            print(f"  {YELLOW}Convención esperada: #.#-iniciales-descripcion.ipynb{RESET}")
        
        if correct:
            self.results["passed"] += 1
            self._log_success(f"✓ {len(correct)} notebooks con convención correcta")
            return True
        
        return False
    
    def check_readme_content(self) -> bool:
        """Verifica contenido mínimo del README."""
        self.results["total_checks"] += 1
        readme = self.project_root / "README.md"
        
        if not readme.exists():
            self.results["failed"] += 1
            self._log_error("✗ README.md NO ENCONTRADO")
            return False
        
        with open(readme, 'r', encoding='utf-8') as f:
            content = f.read().lower()
        
        required_sections = [
            ("project", "descripción del proyecto"),
            ("install", "instrucciones de instalación"),
            ("usage", "instrucciones de uso"),
            ("structure", "estructura del proyecto"),
        ]
        
        missing = []
        for keyword, description in required_sections:
            if keyword not in content:
                missing.append(description)
        
        if missing:
            self.results["warnings"] += 1
            self._log_warning("⚠ README.md podría mejorar con estas secciones:")
            for section in missing:
                print(f"    - {section}")
            return False
        
        self.results["passed"] += 1
        self._log_success("✓ README.md tiene las secciones principales")
        return True
    
    def _log_success(self, message: str):
        """Imprime mensaje de éxito."""
        print(f"{GREEN}{message}{RESET}")
        self.results["details"].append(("success", message))
    
    def _log_error(self, message: str):
        """Imprime mensaje de error."""
        print(f"{RED}{message}{RESET}")
        self.results["details"].append(("error", message))
    
    def _log_warning(self, message: str):
        """Imprime mensaje de advertencia."""
        print(f"{YELLOW}{message}{RESET}")
        self.results["details"].append(("warning", message))
    
    def _log_info(self, message: str):
        """Imprime mensaje informativo."""
        print(f"{BLUE}{message}{RESET}")
    
    def validate_all(self, module_name: str = "acoustic_ml") -> Dict:
        """Ejecuta todas las validaciones."""
        print(f"\n{BOLD}{'='*70}{RESET}")
        print(f"{BOLD}{BLUE}VALIDACIÓN COOKIECUTTER DATA SCIENCE - FASE 2{RESET}")
        print(f"{BOLD}{'='*70}{RESET}\n")
        
        print(f"{BOLD}Proyecto:{RESET} {self.project_root}")
        print(f"{BOLD}Módulo Python:{RESET} {module_name}\n")
        
        # 1. Estructura de directorios principal
        print(f"\n{BOLD}[1] ESTRUCTURA DE DIRECTORIOS{RESET}")
        print("-" * 50)
        self.check_directory("data", "Directorio data")
        self.check_directory("data/raw", "Directorio data/raw")
        self.check_directory("data/processed", "Directorio data/processed")
        self.check_directory("data/interim", "Directorio data/interim", required=False)
        self.check_directory("data/external", "Directorio data/external", required=False)
        
        self.check_directory("models", "Directorio models")
        self.check_directory("notebooks", "Directorio notebooks")
        self.check_directory("reports", "Directorio reports")
        self.check_directory("reports/figures", "Directorio reports/figures")
        self.check_directory("references", "Directorio references", required=False)
        
        # 2. Archivos de configuración
        print(f"\n{BOLD}[2] ARCHIVOS DE CONFIGURACIÓN{RESET}")
        print("-" * 50)
        self.check_file("README.md", "README principal")
        self.check_file("requirements.txt", "requirements.txt", required=False)
        self.check_file("pyproject.toml", "pyproject.toml", required=False)
        self.check_file(".gitignore", ".gitignore")
        self.check_file("Makefile", "Makefile", required=False)
        
        # 3. Control de versiones
        print(f"\n{BOLD}[3] CONTROL DE VERSIONES{RESET}")
        print("-" * 50)
        self.check_git_repo()
        self.check_dvc_config()
        self.check_file(".dvcignore", ".dvcignore", required=False)
        
        # 4. Módulo Python
        print(f"\n{BOLD}[4] MÓDULO PYTHON: {module_name}{RESET}")
        print("-" * 50)
        self.check_python_module(module_name)
        
        # 5. Notebooks
        print(f"\n{BOLD}[5] NOTEBOOKS{RESET}")
        print("-" * 50)
        self.check_notebooks_naming()
        
        # 6. README content
        print(f"\n{BOLD}[6] DOCUMENTACIÓN{RESET}")
        print("-" * 50)
        self.check_readme_content()
        
        # Resumen final
        self._print_summary()
        
        return self.results
    
    def _print_summary(self):
        """Imprime resumen de resultados."""
        print(f"\n{BOLD}{'='*70}{RESET}")
        print(f"{BOLD}RESUMEN DE VALIDACIÓN{RESET}")
        print(f"{BOLD}{'='*70}{RESET}\n")
        
        total = self.results["total_checks"]
        passed = self.results["passed"]
        failed = self.results["failed"]
        warnings = self.results["warnings"]
        
        print(f"Total de verificaciones: {BOLD}{total}{RESET}")
        print(f"{GREEN}✓ Exitosas: {passed}{RESET}")
        print(f"{RED}✗ Fallidas: {failed}{RESET}")
        print(f"{YELLOW}⚠ Advertencias: {warnings}{RESET}\n")
        
        percentage = (passed / total * 100) if total > 0 else 0
        
        if percentage == 100:
            print(f"{GREEN}{BOLD}🎉 ¡EXCELENTE! Tu proyecto está 100% alineado con Cookiecutter Data Science{RESET}")
        elif percentage >= 80:
            print(f"{BLUE}{BOLD}✓ Muy bien! Tu proyecto está {percentage:.1f}% alineado con Cookiecutter{RESET}")
            print(f"{YELLOW}  Considera resolver las advertencias para mejorar{RESET}")
        elif percentage >= 60:
            print(f"{YELLOW}{BOLD}⚠ Tu proyecto está {percentage:.1f}% alineado con Cookiecutter{RESET}")
            print(f"{YELLOW}  Es recomendable resolver los problemas identificados{RESET}")
        else:
            print(f"{RED}{BOLD}✗ Tu proyecto necesita mejoras ({percentage:.1f}% alineado){RESET}")
            print(f"{RED}  Revisa los errores y reestructura según Cookiecutter Data Science{RESET}")
        
        print(f"\n{BOLD}{'='*70}{RESET}\n")


def main():
    """Función principal."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Valida estructura Cookiecutter Data Science"
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=".",
        help="Ruta raíz del proyecto (default: directorio actual)"
    )
    parser.add_argument(
        "--module",
        type=str,
        default="acoustic_ml",
        help="Nombre del módulo Python (default: acoustic_ml)"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Exportar resultados en formato JSON"
    )
    
    args = parser.parse_args()
    
    project_root = Path(args.project_root).resolve()
    
    if not project_root.exists():
        print(f"{RED}Error: El directorio '{project_root}' no existe{RESET}")
        sys.exit(1)
    
    validator = CookiecutterValidator(project_root)
    results = validator.validate_all(module_name=args.module)
    
    if args.json:
        json_output = project_root / "cookiecutter_validation_report.json"
        with open(json_output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n{GREEN}Reporte JSON guardado en: {json_output}{RESET}")
    
    # Exit code basado en resultados
    if results["failed"] > 0:
        sys.exit(1)
    elif results["warnings"] > 0:
        sys.exit(0)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
