@echo off
cd /d "%~dp0"

echo ================================
echo  Avvio AI Oracle API (Windows)
echo ================================

REM [1/3] Activate virtual environment
echo [1/3] Activating virtual environment 'ai-oracle'...
CALL "D:\Users\Utente1\miniconda3\Scripts\activate.bat" ai-oracle

REM [2/3] Check for .env file
echo [2/3] Checking .env file in project's root...
IF NOT EXIST .env (
    echo ERROR: .env not found in project's root!
    pause
    exit /b 1
)

REM [3/3] Start API
echo [3/3] Start API with Uvicorn...
uvicorn scripts.inference_api:app --host 0.0.0.0 --port 8000 --reload

pause
