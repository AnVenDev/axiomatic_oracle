@echo off
setlocal
cls
echo ================================
echo   Test Suite - AI Oracle RWA
echo ================================

REM [1/5] Attiva env
echo [1/5] Attivazione ambiente 'ai-oracle'...
CALL D:\Users\Utente1\miniconda3\condabin\conda.bat activate ai-oracle
IF ERRORLEVEL 1 (
  echo ERRORE: impossibile attivare l'ambiente conda 'ai-oracle'.
  pause
  exit /b 1
)

REM [2/5] Env e percorsi
set PYTHONPATH=%CD%
IF EXIST "shared\outputs\models" (
  set AI_ORACLE_MODELS_BASE=%CD%\shared\outputs\models
) ELSE (
  set AI_ORACLE_MODELS_BASE=%CD%\models
)
echo MODELS_BASE = %AI_ORACLE_MODELS_BASE%

REM [3/5] Verifica che l'API sia su (health check)
echo [3/5] Verifica API /health ...
powershell -NoProfile -Command ^
  "$u='http://127.0.0.1:8000/health'; $ok=$false; for($i=0;$i -lt 20;$i++){try{$r=Invoke-WebRequest -UseBasicParsing $u -TimeoutSec 2; if($r.StatusCode -eq 200){$ok=$true; break}}catch{}; Start-Sleep -Milliseconds 500}; if(-not $ok){Write-Error 'API non raggiungibile. Avviala con start.bat in un\'altra finestra.'; exit 1}"
IF ERRORLEVEL 1 (
  echo ERRORE: API non raggiungibile su 127.0.0.1:8000. Avviala con start.bat e riprova.
  pause
  exit /b 1
)

REM [4/5] E2E sanity (richiede API up, NON pubblica su testnet)
echo.
echo [4/5] Test end-to-end (sanity) ...
python tests\test_e2e_sanity_check.py
IF ERRORLEVEL 1 (
  echo Errore in test_e2e_sanity_check.py
)

REM [5/5] Pytest unit/integration
echo.
echo [5/5] Test con Pytest (unit + integrazione) ...
pytest -q ^
  tests\test_secrets_manager.py ^
  tests\test_logger_utils.py ^
  tests\test_sample_property.py ^
  tests\test_model_registry.py ^
  tests\test_algorand_utils.py ^
  tests\test_blockchain_publish.py ^
  tests\test_api.py

IF ERRORLEVEL 1 (
  echo Alcuni test sono falliti.
) ELSE (
  echo Tutti i test sono PASSATI.
)

echo.
echo ================================
echo   Fine test suite
echo ================================
endlocal
pause