@echo off
setlocal
cd /d "%~dp0"

echo ================================
echo   Avvio AI Oracle API (Windows)
echo ================================

REM [1/4] Activate virtual environment
echo [1/4] Activating conda env 'ai-oracle'...
CALL "D:\Users\Utente1\miniconda3\Scripts\activate.bat" ai-oracle
IF ERRORLEVEL 1 (
  echo ERRORE: impossibile attivare l'ambiente conda 'ai-oracle'.
  pause
  exit /b 1
)

REM [2/4] Project root as PYTHONPATH
set PYTHONPATH=%CD%

REM [3/4] Modelli: usa gli artifacts prodotti dai notebooks
REM       (shared\outputs\models\property\value_regressor_v*.joblib)
set AI_ORACLE_MODELS_BASE=%CD%\notebooks\outputs\modeling\artifacts

echo MODELS_BASE = %AI_ORACLE_MODELS_BASE%

REM [4/4] Start API
echo [4/4] Starting Uvicorn on http://127.0.0.1:8000 ...
uvicorn scripts.inference_api:app --host 127.0.0.1 --port 8000 --reload
REM Nota: uvicorn resta in foreground in questa finestra

endlocal
pause