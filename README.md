# MLOps Equipo 24 – Music Emotion Recognition

---

## 📚 Información académica

**Instituto Tecnológico y de Estudios Superiores de Monterrey**  
**Maestría en Inteligencia Artificial Aplicada (MNA)**  
**Curso:** Operaciones de aprendizaje automático  
**Actividad:** Asistir a Sesión de Integración de tu equipo de proyecto  
**Tema:** *Uso de Discord para comunicaciones del curso*  

**Integrantes del equipo N° 24:**
- A01796937 – Sandra Luz Cervantes Espinoza  
- A01226881 – Héctor Jesús López Meza  
- A01796697 – Mauricio Torres Baena  
- A01360416 – David Cruz Beltrán  
- A01795838 – Javier Augusto Rebull Saucedo  

**Profesores:**  
- Profesor Titular: Dr. Gerardo Rodríguez Hernández  
- Profesor Titular: Maestro Ricardo Valdez Hernández  
- Profesor Asistente: Maestra María Mylen Treviño Elizondo  
- Profesor Tutor: José Ángel Martínez Navarro  

📅 **Periodo:** Septiembre a Diciembre 2025  

---

## 📦 Proyecto

Este repositorio contiene notebooks, scripts y artefactos de MLflow relacionados con el proyecto.  
Los datos y modelos están versionados con DVC y almacenados en un bucket de S3.

---

## 🚀 Setup inicial
1. Clona el repositorio

```bash
git clone https://github.com/jrebull/MLOps_Team24.git
cd MLOps_Team24

	2.	Crea un entorno virtual e instala dependencias

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

	3.	Configura tus credenciales de AWS (solo la primera vez)
Debes tener un archivo ~/.aws/credentials con este formato:

[default]
aws_access_key_id = TU_ACCESS_KEY_ID
aws_secret_access_key = TU_SECRET_ACCESS_KEY
region = us-east-1


⸻

📦 Descargar datasets y modelos

Para obtener los datos desde el bucket S3:

dvc pull

Esto descargará Acoustic Features.csv y cualquier otro artefacto versionado.

⸻

📒 Trabajar con notebooks
	•	Con Jupyter Lab:

jupyter-lab

	•	Con VSCode:

code .


⸻

📈 Tracking de experimentos con MLflow
	1.	Levanta el servidor MLflow en local:

mlflow ui --port 5001

	2.	Abre en tu navegador: http://127.0.0.1:5001

⸻

🔄 Reproducir el pipeline

Para ejecutar el pipeline y generar métricas:

# Ejecuta todas las etapas definidas en dvc.yaml
dvc repro

# Compara métricas actuales contra la última versión en Git
dvc metrics diff

Esto permite ver cómo evolucionan las métricas (accuracy, F1, etc.) entre corridas y commits.

⸻

👩‍💻 Flujo de contribución
	1.	Crea una nueva rama para tu contribución:

git checkout -b feat/<nombre-de-tu-rama>

	2.	Asegúrate de correr MLflow en tu máquina.
	3.	Realiza cambios y, si generas datos/modelos, súbelos a DVC:

dvc add <ruta-al-archivo>
git add <ruta-al-archivo>.dvc
git commit -m "Agrega datos/modelos a DVC"
git push origin feat/<nombre-de-tu-rama>
dvc push

	4.	Haz un Pull Request a main.

---
