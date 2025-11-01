#!/bin/bash
# Setup Script: Angry Analysis Package
# =====================================
# Este script instala y configura el paquete de análisis completo

set -e  # Exit on error

echo "🚀 MLOps Team 24: Angry Analysis Package Setup"
echo "=============================================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 1. Verificar que estamos en el directorio correcto
echo "📂 Verificando directorio..."
if [ ! -f "data/processed/turkish_music_emotion_v2_cleaned_full.csv" ]; then
    echo -e "${RED}❌ ERROR: No se encontró el dataset${NC}"
    echo "Por favor ejecuta este script desde:"
    echo "  /Users/haowei/Documents/MLOps/MNA_Team24/MLOps_Team24"
    exit 1
fi
echo -e "${GREEN}✅ Directorio correcto${NC}"
echo ""

# 2. Verificar Python
echo "🐍 Verificando Python..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo -e "${GREEN}✅ $PYTHON_VERSION${NC}"
else
    echo -e "${RED}❌ Python3 no encontrado${NC}"
    exit 1
fi
echo ""

# 3. Verificar/Activar virtual environment
echo "📦 Verificando virtual environment..."
if [ -d ".venv" ]; then
    echo -e "${GREEN}✅ Virtual environment encontrado${NC}"
    echo "   Activando .venv..."
    source .venv/bin/activate
else
    echo -e "${YELLOW}⚠️  Virtual environment no encontrado${NC}"
    echo "   ¿Crear uno? (y/n): "
    read -r response
    if [ "$response" = "y" ]; then
        python3 -m venv .venv
        source .venv/bin/activate
        echo -e "${GREEN}✅ Virtual environment creado${NC}"
    else
        echo "   Continuando sin virtual environment..."
    fi
fi
echo ""

# 4. Verificar dependencias
echo "📚 Verificando dependencias..."
REQUIRED_PACKAGES=("pandas" "numpy" "scikit-learn" "mlflow" "librosa" "matplotlib" "seaborn")
MISSING_PACKAGES=()

for package in "${REQUIRED_PACKAGES[@]}"; do
    if python3 -c "import $package" 2>/dev/null; then
        echo -e "  ${GREEN}✅${NC} $package"
    else
        echo -e "  ${RED}❌${NC} $package"
        MISSING_PACKAGES+=("$package")
    fi
done

if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
    echo ""
    echo -e "${YELLOW}⚠️  Paquetes faltantes: ${MISSING_PACKAGES[*]}${NC}"
    echo "   ¿Instalar ahora? (y/n): "
    read -r response
    if [ "$response" = "y" ]; then
        pip install "${MISSING_PACKAGES[@]}"
        echo -e "${GREEN}✅ Dependencias instaladas${NC}"
    fi
fi
echo ""

# 5. Verificar acoustic_ml
echo "🎵 Verificando acoustic_ml..."
if python3 -c "from acoustic_ml.features.audio_features import AudioFeatureExtractor" 2>/dev/null; then
    echo -e "${GREEN}✅ acoustic_ml instalado${NC}"
else
    echo -e "${YELLOW}⚠️  acoustic_ml no encontrado${NC}"
    if [ -f "setup.py" ]; then
        echo "   ¿Instalar acoustic_ml? (y/n): "
        read -r response
        if [ "$response" = "y" ]; then
            pip install -e .
            echo -e "${GREEN}✅ acoustic_ml instalado${NC}"
        fi
    fi
fi
echo ""

# 6. Verificar MLflow
echo "🔬 Verificando MLflow..."
if [ -d "mlruns" ]; then
    echo -e "${GREEN}✅ MLflow tracking directory encontrado${NC}"
    
    # Verificar que el run_id existe
    RUN_ID="eb05c7698f12499b86ed35ca6efc15a7"
    if find mlruns -type d -name "$RUN_ID" | grep -q .; then
        echo -e "${GREEN}✅ Run ID $RUN_ID encontrado${NC}"
    else
        echo -e "${YELLOW}⚠️  Run ID $RUN_ID no encontrado${NC}"
        echo "   El análisis continuará, pero algunos scripts pueden fallar"
    fi
else
    echo -e "${RED}❌ MLflow tracking directory no encontrado${NC}"
fi
echo ""

# 7. Verificar sample_audio
echo "🎵 Verificando sample_audio..."
if [ -d "turkish_music_app/assets/sample_audio" ]; then
    AUDIO_COUNT=$(find turkish_music_app/assets/sample_audio -name "*.mp3" | wc -l)
    echo -e "${GREEN}✅ Sample audio directory encontrado${NC}"
    echo "   Archivos .mp3 encontrados: $AUDIO_COUNT"
else
    echo -e "${YELLOW}⚠️  Sample audio directory no encontrado${NC}"
    echo "   Script 3 (sample_audio_features) no funcionará"
fi
echo ""

# 8. Hacer scripts ejecutables
echo "⚙️  Configurando permisos..."
chmod +x analyze_*.py run_complete_analysis.py
echo -e "${GREEN}✅ Scripts configurados${NC}"
echo ""

# 9. Resumen
echo "=============================================="
echo "📊 RESUMEN DE SETUP"
echo "=============================================="
echo ""
echo "Archivos instalados:"
echo "  ✅ analyze_1_dataset_distribution.py"
echo "  ✅ analyze_2_confusion_matrix.py"
echo "  ✅ analyze_3_sample_audio_features.py"
echo "  ✅ analyze_4_feature_importance.py"
echo "  ✅ run_complete_analysis.py"
echo "  ✅ README_ANALYSIS.md"
echo "  ✅ DIAGNOSTIC_CHECKLIST.md"
echo "  ✅ QUICK_START.md"
echo ""
echo "=============================================="
echo "🎯 PRÓXIMO PASO"
echo "=============================================="
echo ""
echo "Para ejecutar el análisis completo:"
echo ""
echo -e "  ${GREEN}python3 run_complete_analysis.py${NC}"
echo ""
echo "O ejecutar scripts individuales:"
echo ""
echo "  python3 analyze_1_dataset_distribution.py"
echo "  python3 analyze_2_confusion_matrix.py"
echo "  python3 analyze_3_sample_audio_features.py"
echo "  python3 analyze_4_feature_importance.py"
echo ""
echo "Para más información:"
echo "  cat README_ANALYSIS.md"
echo "  cat QUICK_START.md"
echo ""
echo -e "${GREEN}✅ Setup completado exitosamente!${NC}"
echo ""
