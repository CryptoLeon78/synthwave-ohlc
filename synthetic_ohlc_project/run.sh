#!/usr/bin/env bash
# ============================================================
# run.sh — Script de arranque para SyntheticData QXPRO vQDemy
# ============================================================
# Uso:
#   chmod +x run.sh
#   ./run.sh
#
# Opciones:
#   ./run.sh --install    Solo instala dependencias
#   ./run.sh --port 8502  Usar puerto específico
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PORT=8501

# Parsear argumentos
INSTALL_ONLY=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --install)
            INSTALL_ONLY=true
            shift
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

echo "============================================"
echo "  SyntheticData QXPRO vQDemy"
echo "  Generador de Data Sintética OHLC"
echo "============================================"
echo ""

# Verificar Python
if command -v python3 &> /dev/null; then
    PYTHON=python3
elif command -v python &> /dev/null; then
    PYTHON=python
else
    echo "❌ Python no encontrado. Instala Python 3.8+ primero."
    exit 1
fi

echo "🐍 Usando: $($PYTHON --version)"

# Crear entorno virtual si no existe
if [ ! -d "venv" ]; then
    echo "📦 Creando entorno virtual..."
    $PYTHON -m venv venv
fi

# Activar entorno virtual
echo "🔧 Activando entorno virtual..."
source venv/bin/activate

# Instalar dependencias
echo "📥 Instalando dependencias..."
pip install -q -r requirements.txt

if [ "$INSTALL_ONLY" = true ]; then
    echo "✅ Dependencias instaladas correctamente."
    exit 0
fi

echo ""
echo "🚀 Iniciando aplicación en http://localhost:$PORT"
echo "   Presiona Ctrl+C para detener."
echo ""

streamlit run app.py --server.port $PORT --server.headless true
