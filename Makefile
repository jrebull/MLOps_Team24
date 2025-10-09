# ========================
# Makefile - MLOps Equipo 24
# ========================

# Variables
PYTHON=python
VENV=.venv
PORT=5001

# ========================
# Setup del entorno
# ========================

setup:
	$(PYTHON) -m venv $(VENV)
	. $(VENV)/bin/activate && pip install -U pip && pip install -r requirements.txt

# ========================
# Jupyter / Notebooks
# ========================

jupyter:
	. $(VENV)/bin/activate && jupyter lab

# ========================
# MLflow Tracking
# ========================

mlflow:
	. $(VENV)/bin/activate && mlflow ui --port $(PORT)

# ========================
# DVC / Pipeline
# ========================

pull:
	. $(VENV)/bin/activate && dvc pull

push:
	. $(VENV)/bin/activate && dvc push

train:
	. $(VENV)/bin/activate && dvc repro -f train

reproduce:
	. $(VENV)/bin/activate && dvc repro

metrics:
	. $(VENV)/bin/activate && dvc metrics show

diff:
	. $(VENV)/bin/activate && dvc metrics diff

status:
	. $(VENV)/bin/activate && dvc status

# ========================
# Limpieza
# ========================

clean:
	rm -rf $(VENV) .ipynb_checkpoints __pycache__ .pytest_cache
	find . -type d -name '*.egg-info' -exec rm -rf {} +