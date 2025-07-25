@echo off
cls
echo ================================
echo  Test Suite - AI Oracle RWA
echo ================================

REM Step 1: Attivazione ambiente Conda
echo [1/3] Attivazione ambiente 'ai-oracle'...
CALL D:\Users\Utente1\miniconda3\condabin\conda.bat activate ai-oracle

REM Step 2: Esecuzione test e2e
echo.
echo [2/3] Test end-to-end (test_e2e_sanity_check.py)...
python tests/test_e2e_sanity_check.py
IF %ERRORLEVEL% NEQ 0 (
    echo Errore in test_e2e_sanity_check.py
)

REM Step 3: Esecuzione Pytest su API e Publisher
echo.
echo [3/3] Test con Pytest (API e Publisher)...
pytest tests/test_api.py
IF %ERRORLEVEL% NEQ 0 (
    echo Errore nei test_api
)

pytest tests/test_blockchain_publish.py
IF %ERRORLEVEL% NEQ 0 (
    echo Errore nei test_blockchain_publish
)

echo.
echo ================================
echo  Fine test suite
echo ================================
pause
