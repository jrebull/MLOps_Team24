# 🔥 SMOKE TESTS - RESUMEN EJECUTIVO

## ¿QUÉ ES?

Conjunto de **scripts de validación profesionales** que verifican que TODOS los servicios y endpoints funcionan correctamente después de hacer `docker compose up`.

**Objetivo:** Asegurar que la aplicación está lista para producción antes de presentar al profesor.

---

## 📦 ARCHIVOS GENERADOS

| Archivo | Descripción | Usar para |
|---------|-------------|----------|
| `smoke_tests.py` | Script principal con 13 tests | Validación completa post-deploy |
| `quick_diagnostics.py` | Diagnóstico rápido sin esperas | Diagnóstico rapido de problemas |
| `run_smoke_tests.sh` | Script Bash wrapper | Ejecución simple (chmod +x primero) |
| `SMOKE_TESTS_README.md` | Documentación detallada | Referencia de uso |
| `SMOKE_TESTS_WORKFLOW.md` | Guía de workflow completo | Integración en CI/CD |
| `Makefile.smoke_tests` | Targets para Makefile | Agregar a tu Makefile |

---

## 🚀 QUICK START (5 MIN)

### PASO 1: Copiar archivos

```bash
cd /Users/haowei/Documents/MLOps/MNA_Team24/MLOps_Team24

# Copiar scripts
cp smoke_tests.py .
cp quick_diagnostics.py .
cp run_smoke_tests.sh .

# Hacer script ejecutable
chmod +x run_smoke_tests.sh
```

### PASO 2: Levantar servicios

```bash
# En una terminal
docker compose up

# Espera 30-40 segundos hasta que ambos servicios estén listos
```

### PASO 3: Ejecutar smoke tests

```bash
# En otra terminal
python smoke_tests.py

# O con script bash
./run_smoke_tests.sh
```

### PASO 4: Revisar resultados

```bash
# Si todo paso:
✅ ALL TESTS PASSED!

# Reporte JSON generado:
cat smoke_test_report.json
```

---

## 🎯 QUÉ VALIDA

**13 tests profesionales** que verifican:

```
FastAPI:
  ✓ Root endpoint
  ✓ Health check
  ✓ API documentation (Swagger)

Endpoints:
  ✓ Predict (predicción con features)
  ✓ List models
  ✓ JSON response format

MLflow:
  ✓ Server connectivity
  ✓ API endpoint

Validación:
  ✓ Schema validation (Pydantic v2)
  ✓ Error handling (404, 422)

ML:
  ✓ Model loading
  ✓ Concurrent requests
```

---

## 📊 FLUJO TÍPICO

```
1. git commit cambios
2. docker compose up
3. python smoke_tests.py
   
   ├─ SI PASA ✅
   │  └─ Listo para presentar
   │
   └─ SI FALLA ❌
      ├─ python quick_diagnostics.py (diagnóstico rápido)
      ├─ docker compose logs web (ver errores)
      └─ Fijar y volver a paso 3
```

---

## 📈 MÉTRICAS DE ÉXITO

```
Total Tests: 13
Passed: 13 ✓
Failed: 0 ✗
Pass Rate: 100%
Time: ~5-10 segundos

Estado: ✅ LISTO PARA PRODUCCIÓN
```

---

## 🔧 DIAGNÓSTICO RÁPIDO (SI FALLA)

Si algún test falla, ejecuta esto primero:

```bash
# Diagnostico rápido - muestra estado de servicios en 5 seg
python quick_diagnostics.py

# Output esperado:
# OK   | FastAPI Root           | http://127.0.0.1:8000/
# OK   | FastAPI Health         | http://127.0.0.1:8000/api/v1/health
# OK   | Swagger Docs           | http://127.0.0.1:8000/docs
# OK   | MLflow Server          | http://127.0.0.1:5001/
# OK   | MLflow API             | http://127.0.0.1:5001/api/2.0/health
# OK   | Predict works          | Prediction: Happy
```

---

## 🛠️ INSTALACIÓN COMPLETA (PASO A PASO)

### EN TU MÁQUINA

```bash
# 1. Navega al proyecto
cd /Users/haowei/Documents/MLOps/MNA_Team24/MLOps_Team24

# 2. Copiar archivos
cp /ruta/a/archivos/smoke_tests.py .
cp /ruta/a/archivos/quick_diagnostics.py .
cp /ruta/a/archivos/run_smoke_tests.sh .

# 3. Hacer ejecutable
chmod +x run_smoke_tests.sh

# 4. Agregar al Makefile (opcional pero recomendado)
cat Makefile.smoke_tests >> Makefile
# Verifica que se agregó:
tail -20 Makefile

# 5. Verificar requirements tienen 'requests'
grep requests requirements-prod.txt
# Si no está, agregar:
echo "requests>=2.31.0" >> requirements-prod.txt

# 6. Commit a Git
git add smoke_tests.py quick_diagnostics.py run_smoke_tests.sh
git commit -m "chore: agregar smoke tests post-deploy"
git push origin main
```

### PRIMERA EJECUCIÓN

```bash
# 1. Levantar servicios
docker compose up -d

# 2. Esperar servicios (IMPORTANTE!)
sleep 40

# 3. Ejecutar tests
python smoke_tests.py

# 4. Esperar output completo (toma ~5-10 seg)
```

---

## ✅ CHECKLIST DE SETUP

```bash
□ Archivos copiados al proyecto
□ run_smoke_tests.sh tiene permisos executable (chmod +x)
□ requirements-prod.txt tiene 'requests'
□ docker-compose.yml en versión correcta
□ config.env.example existe
□ Modelo en app/models/model.joblib existe
□ pytest tests/ -v pasa (33/33 tests)
```

---

## 🚀 USAR EN PRODUCCIÓN

### Deploy manual

```bash
# En servidor o máquina de deployment
git clone <repo>
cd MLOps_Team24
cp config.env.example config.env  # Editar con valores reales

docker compose up -d
sleep 40
python smoke_tests.py

# Si pasa: ✅ DONE
# Si falla: Ver logs y diagnosticar
```

### Deploy automático (CI/CD)

En tu workflow de GitHub Actions (`.github/workflows/deploy.yml`):

```yaml
- name: Run Smoke Tests
  run: |
    pip install requests
    python smoke_tests.py
    
- name: Upload Report
  if: always()
  uses: actions/upload-artifact@v3
  with:
    name: smoke-test-report
    path: smoke_test_report.json
```

---

## 🔍 TROUBLESHOOTING RÁPIDO

| Problema | Solución |
|----------|----------|
| "Timeout esperando FastAPI" | `docker compose logs web` \| vé errores |
| "Predict endpoint FAIL" | Modelo falta: `docker compose exec web ls app/models/model.joblib` |
| "MLflow FAIL" | MLflow no levanta: `docker compose logs mlflow` |
| "Schema validation FAIL" | Verifica Pydantic v2: `docker compose exec web pip list \| grep pydantic` |
| "No module requests" | Instala: `pip install requests` |

---

## 📝 PRÓXIMOS PASOS RECOMENDADOS

1. ✅ **Hoy:** Copiar archivos y probar localmente
   ```bash
   docker compose up && python smoke_tests.py
   ```

2. ✅ **Commit:** Agregar a Git
   ```bash
   git add smoke_tests.py quick_diagnostics.py run_smoke_tests.sh
   git commit -m "chore: agregar smoke tests"
   ```

3. ✅ **Documentación:** Agregar sección a README principal
   ```markdown
   ## Validación Post-Deploy
   
   Después de `docker compose up`, ejecuta:
   ```bash
   python smoke_tests.py
   ```
   Todos los 13 tests deben pasar.
   ```

4. ✅ **Presentación:** Mostrar smoke tests al profesor
   - "Aquí ejecuto smoke tests para validar todo funciona..."
   - Mostrar output completo
   - Mostrar reporte JSON

---

## 💡 PRO TIPS

```bash
# Ver solo fallos (si los hay)
python smoke_tests.py | grep FAIL

# Grabar output con timestamp
python smoke_tests.py | tee smoke_test_$(date +%Y%m%d_%H%M%S).log

# Ejecutar con Makefile (si lo agregaste)
make smoke-tests

# Diagnóstico rápido en 5 segundos
python quick_diagnostics.py

# Watch de docker logs mientras corre
docker compose logs -f web &
python smoke_tests.py
```

---

## 🎓 PARA EL PROFESOR

Cuando presentes Phase 3, muestra esto:

```bash
# 1. Mostrar que Docker levanta limpio
docker compose up -d
sleep 40

# 2. Ejecutar smoke tests
python smoke_tests.py

# 3. Mostrar reporte
cat smoke_test_report.json | jq .

# 4. Explicar:
"Estos 13 smoke tests validan que:
 - FastAPI levanta sin errores
 - Todos los endpoints responden correctamente
 - El modelo se carga y predice
 - Validación de schema funciona
 - Manejo de errores es robusto
 
 Pass rate: 100% = Producción-ready"
```

---

## 📞 HELP

Si algo no funciona:

1. Lee: `SMOKE_TESTS_README.md`
2. Revisa: `docker compose logs`
3. Test manual:
   ```bash
   curl http://127.0.0.1:8000/api/v1/health
   curl http://127.0.0.1:5001/
   ```
4. Diagnóstico:
   ```bash
   python quick_diagnostics.py
   ```

---

## 📦 ARCHIVOS FINALES A COPIAR

```
src/
├── smoke_tests.py              ← COPIAR
├── quick_diagnostics.py         ← COPIAR
├── run_smoke_tests.sh           ← COPIAR
├── Makefile.smoke_tests         ← LEER E INTEGRAR AL MAKEFILE
├── SMOKE_TESTS_README.md        ← REFERENCIA
├── SMOKE_TESTS_WORKFLOW.md      ← REFERENCIA
└── Este archivo                 ← GUÍA
```

---

## 🎯 ÉXITO

Cuando veas esto:

```
🎉 ALL TESTS PASSED!
Pass Rate: 100.0%
Time: 5.42s
```

**Significa:** ✅ Phase 3 está lista para presentar

---

**Versión:** 1.0
**Fecha:** 2025-11-12
**Para:** MLOps Team24 - Turkish Music Emotion Recognition
