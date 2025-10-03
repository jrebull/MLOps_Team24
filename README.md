⸻

🎵 MLOps Equipo 24 – Music Emotion Recognition

Este repositorio contiene notebooks, scripts y artefactos de MLflow relacionados con el proyecto.
Los datos y modelos están versionados con DVC y almacenados en un bucket de S3.

⸻

📘 Información académica

Instituto Tecnológico y de Estudios Superiores de Monterrey
Maestría en Inteligencia Artificial Aplicada (MNA)
Curso: Operaciones de Aprendizaje Automático
Actividad: Asistir a Sesión de Integración de tu equipo de proyecto

Tema: Uso de Discord para comunicaciones del curso

Integrantes del equipo N° 24:
	•	A01796937 - Sandra Luz Cervantes Espinoza
	•	A01226881 - Héctor Jesús López Meza
	•	A01796697 - Mauricio Torres Baena
	•	A01360416 - David Cruz Beltrán
	•	A01795838 - Javier Augusto Rebull Saucedo

Profesores:
	•	Dr. Gerardo Rodríguez Hernández (Titular)
	•	Maestro Ricardo Valdez Hernández (Titular)
	•	Maestra María Mylen Treviño Elizondo (Asistente)
	•	José Ángel Martínez Navarro (Tutor)

📅 Septiembre – Diciembre 2025

⸻

🚀 Setup inicial
	1.	Clona el repositorio

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

Ahí verás los runs del experimento Equipo24-MER con sus métricas.

⸻

🔄 Reproducir el pipeline con DVC

El pipeline está definido en dvc.yaml.
Para ejecutarlo y regenerar métricas/modelos:

dvc repro

Para ver las métricas actuales:

dvc metrics show

Para comparar métricas entre commits:

dvc metrics diff


⸻

📊 Diagrama del flujo de trabajo

flowchart TD
    A[📂 Dataset: Acoustic Features.csv] -->|dvc add| B[DVC Tracking]
    B -->|almacenado en| C[S3 Bucket]
    A --> D[⚙️ train_baseline.py]
    D --> E[🤖 Modelo entrenado]
    D --> F[📈 metrics.json]
    E -->|log_model| G[MLflow Tracking]
    F -->|log_metrics| G
    G --> H[MLflow UI http://127.0.0.1:5001]

Este diagrama muestra cómo:
	•	Los datos se versionan con DVC y se guardan en S3.
	•	El script train_baseline.py entrena el modelo y genera métricas.
	•	Las métricas y modelos se registran en MLflow, accesibles desde la UI local.

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

