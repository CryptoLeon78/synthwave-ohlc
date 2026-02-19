@echo off
REM ============================================================
REM run.bat — Script de arranque para SyntheticData QXPRO vQDemy
REM ============================================================
REM Uso: doble clic o ejecutar desde cmd

cd /d "%~dp0"

echo ============================================
echo   SyntheticData QXPRO vQDemy
echo   Generador de Data Sintetica OHLC
echo ============================================
echo.

REM Verificar Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python no encontrado. Instala Python 3.8+
    pause
    exit /b 1
)

REM Crear entorno virtual si no existe
if not exist "venv" (
    echo Creando entorno virtual...
    python -m venv venv
)

REM Activar entorno virtual
echo Activando entorno virtual...
call venv\Scripts\activate.bat

REM Instalar dependencias
echo Instalando dependencias...
pip install -q -r requirements.txt

echo.
echo Iniciando aplicacion...
echo.

streamlit run app.py --server.port 8501 --server.headless true

pause
