@echo off
cd /d "%~dp0"

echo ================================
echo  Avvio AI Oracle API (Windows)
echo ================================

REM [1/3] Attivazione ambiente virtuale
echo [1/3] Attivazione ambiente virtuale 'ai-oracle'...
CALL "D:\Users\Utente1\miniconda3\Scripts\activate.bat" ai-oracle

REM [2/3] Controllo file .env
echo [2/3] Controllo file .env nella root del progetto...
IF NOT EXIST .env (
    echo ERRORE: Il file .env non esiste nella root!
    pause
    exit /b 1
)

REM [3/3] Avvia API
echo [3/3] Avvio API con Uvicorn...
uvicorn scripts.inference_api:app --host 0.0.0.0 --port 8000 --reload

pause
