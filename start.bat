@echo off
setlocal
REM ──────────────────────────────────────────────────────────────────────────
REM  AI Oracle API (Windows) - launcher
REM  Avvia da root repo (stessa cartella del .bat)
REM ──────────────────────────────────────────────────────────────────────────

REM [0] Root del progetto
set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%" 1>nul
set "ROOT_DIR=%CD%"

echo ================================
echo   Avvio AI Oracle API (Windows)
echo ================================
echo ROOT_DIR = %ROOT_DIR%
echo.

REM [1/5] Attiva ambiente conda
echo [1/5] Activating conda env 'ai-oracle'...
set "CONDA_ACT=D:\Users\Utente1\miniconda3\Scripts\activate.bat"
if exist "%CONDA_ACT%" (
  CALL "%CONDA_ACT%" ai-oracle
) else (
  CALL conda activate ai-oracle
)
IF ERRORLEVEL 1 (
  echo ERRORE: impossibile attivare l'ambiente 'ai-oracle'.
  popd & pause & exit /b 1
)
echo OK.
echo.

REM [2/5] PYTHONPATH → root + notebooks (per notebooks.shared.*)
set "PYTHONPATH=%ROOT_DIR%;%ROOT_DIR%\notebooks"
echo [2/5] PYTHONPATH = %PYTHONPATH%
echo.

REM [3/5] Modelli / Output / Log / Schemi
REM *** MUST: base dei modelli = ...\outputs\modeling (NON ...\artifacts) ***
set "AI_ORACLE_MODELS_BASE=%ROOT_DIR%\notebooks\outputs\modeling"
set "OUTPUTS_DIR=%ROOT_DIR%\notebooks\outputs"
set "AI_ORACLE_LOG_DIR=%OUTPUTS_DIR%\logs"
set "SCHEMAS_DIR=%ROOT_DIR%\schemas"
echo [3/5] MODELS_BASE  = %AI_ORACLE_MODELS_BASE%
echo       OUTPUTS_DIR  = %OUTPUTS_DIR%
echo       LOG_DIR      = %AI_ORACLE_LOG_DIR%
echo       SCHEMAS_DIR  = %SCHEMAS_DIR%

REM check pipeline esistenza (fail-fast gentile)
if not exist "%AI_ORACLE_MODELS_BASE%\property\value_regressor_v2.joblib" (
  echo WARN: pipeline non trovata a:
  echo       %AI_ORACLE_MODELS_BASE%\property\value_regressor_v2.joblib
  echo       (Esegui nb03 oppure verifica il path AI_ORACLE_MODELS_BASE)
)
echo.

REM [4/5] Sicurezza / limiti / PoVal window / CORS
set "ALLOWED_ORIGINS=http://localhost:4200,http://127.0.0.1:4200"
REM set "API_KEY=metti-qui-una-chiave-segreta"
set "RATE_LIMIT_RPS=5"
set "MAX_BODY_BYTES=262144"
set "NOTE_MAX_BYTES=1024"
set "P1_TS_SKEW_PAST=600"
set "P1_TS_SKEW_FUTURE=120"
echo [CFG] ALLOWED_ORIGINS=%ALLOWED_ORIGINS%
if defined API_KEY echo [CFG] API_KEY=*** (set)
echo [CFG] RATE_LIMIT_RPS=%RATE_LIMIT_RPS%   MAX_BODY_BYTES=%MAX_BODY_BYTES%
echo [CFG] NOTE_MAX_BYTES=%NOTE_MAX_BYTES%   P1_TS_SKEW=[-%P1_TS_SKEW_PAST% ; +%P1_TS_SKEW_FUTURE%] s
echo.

REM === Algorand TestNet endpoints (Algonode) ===
set "ALGO_NETWORK=testnet"

REM algod (API node)
set "ALGOD_SERVER=https://testnet-api.algonode.cloud"
set "ALGOD_PORT=443"
set "ALGOD_TOKEN="

REM indexer (opzionale, ma utile per verifiche)
set "INDEXER_SERVER=https://testnet-idx.algonode.cloud"
set "INDEXER_PORT=443"
set "INDEXER_TOKEN="

REM [5/5] Avvio API (Uvicorn)
echo [5/5] Starting Uvicorn on http://127.0.0.1:8000 ...
python -m uvicorn scripts.inference_api:app --host 127.0.0.1 --port 8000 --reload

popd
endlocal
pause
