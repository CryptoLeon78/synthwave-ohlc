# SyntheticData QXPRO vQDemy

Generador de data sintética OHLC con métodos de bootstrap y evaluación estadística.  
Replica fielmente la lógica de **"Sesión 4: Data Sintética, Ensamblado y Filtros"** (PyRE by Quantdemy).

---

## Tabla de Contenidos

1. [Requisitos Previos](#requisitos-previos)
2. [Instalación Rápida](#instalación-rápida)
3. [Ejecución](#ejecución)
4. [Preparar tu CSV](#preparar-tu-csv)
5. [Guía Paso a Paso de Uso](#guía-paso-a-paso-de-uso)
   - [Paso 1: Cargar datos y configurar partición](#paso-1-cargar-datos-y-configurar-partición)
   - [Paso 2: Vectores Referenciados](#paso-2-vectores-referenciados)
   - [Paso 3: Bootstrap Individual](#paso-3-bootstrap-individual)
   - [Paso 4: Generación Masiva](#paso-4-generación-masiva)
   - [Paso 5: Ensemble](#paso-5-ensemble)
   - [Paso 6: Test Adverso](#paso-6-test-adverso)
   - [Paso 7: Monkey Test](#paso-7-monkey-test)
   - [Paso 8: Análisis Anual y Filtros](#paso-8-análisis-anual-y-filtros)
6. [Estructura del Proyecto](#estructura-del-proyecto)
7. [Descripción de Módulos](#descripción-de-módulos)
8. [Exportación de Datos](#exportación-de-datos)
9. [Parámetros Recomendados](#parámetros-recomendados)
10. [Solución de Problemas](#solución-de-problemas)

---

## Requisitos Previos

- **Python 3.8+** instalado
- **pip** (incluido con Python)
- Un archivo CSV con datos OHLC (Date, Open, High, Low, Close)

---

## Instalación Rápida

### Opción A: Script automático (recomendado)

**Linux / macOS:**
```bash
cd synthetic_ohlc_project
chmod +x run.sh
./run.sh
```

**Windows:**
```
Doble clic en run.bat
```

El script crea un entorno virtual, instala dependencias y lanza la app automáticamente.

### Opción B: Manual

```bash
cd synthetic_ohlc_project
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate.bat     # Windows
pip install -r requirements.txt
```

---

## Ejecución

```bash
streamlit run app.py
```

La app se abrirá en tu navegador en `http://localhost:8501`.

Para usar un puerto diferente:
```bash
./run.sh --port 8502
```

---

## Preparar tu CSV

El archivo CSV debe tener **exactamente** estas columnas:

| Columna | Tipo     | Ejemplo        |
|---------|----------|----------------|
| Date    | fecha    | 2010-01-04     |
| Open    | decimal  | 1.4320         |
| High    | decimal  | 1.4440         |
| Low     | decimal  | 1.4280         |
| Close   | decimal  | 1.4410         |

**Ejemplo:**
```csv
Date,Open,High,Low,Close
2010-01-04,1.4320,1.4440,1.4280,1.4410
2010-01-05,1.4410,1.4500,1.4350,1.4480
```

> Los datos deben estar ordenados cronológicamente. Se admite cualquier activo (EURUSD, SPY, BTC, etc.).

---

## Guía Paso a Paso de Uso

### Paso 1: Cargar datos y configurar partición

1. En la **barra lateral izquierda**, haz clic en "Subir CSV con OHLC" y selecciona tu archivo.
2. Configura las **fechas de corte**:
   - **Fin In-Sample**: fecha hasta la cual se entrenarán los modelos (ej: `2018-12-31`).
   - **Fin Out-of-Sample**: fecha hasta la cual se testea fuera de muestra (ej: `2021-12-31`).
   - Todo lo posterior será **Forward**.
3. Selecciona la **columna de precio** para métricas (por defecto `Close`).
4. Elige cómo manejar valores faltantes: `drop`, `interpolate` o `fill_forward`.

> Verás un resumen con el número de filas de cada partición (IS / OOS / Forward).

---

### Paso 2: Vectores Referenciados

**Pestaña: 📊 Vectores Referenciados**

Aquí se convierten los datos OHLC a **vectores referenciados al close anterior**:
- `open_relative`: (Open - Close_anterior) / Close_anterior
- `high_relative`: (High - Close_anterior) / Close_anterior
- `low_relative`: (Low - Close_anterior) / Close_anterior
- `close_relative`: (Close - Close_anterior) / Close_anterior

**Qué verás:**
- Tabla con estadísticas descriptivas (media, std, skew, kurtosis).
- Histogramas de los 4 componentes referenciados.

> Este paso es **obligatorio** antes de generar sintéticas. Se ejecuta automáticamente al cargar datos.

---

### Paso 3: Bootstrap Individual

**Pestaña: 🔄 Bootstrap Individual**

Genera **una sola** serie sintética para inspección rápida.

1. Selecciona el **método de bootstrap**:
   - `simple`: re-muestreo aleatorio con reemplazo.
   - `block`: bloques consecutivos aleatorios (preserva dependencia local).
   - `intra_block`: baraja dentro de cada bloque, bloques fijos.
   - `hybrid`: conserva un % de observaciones originales, reemplaza el resto.
   - `block_aware`: como hybrid pero opera por bloques completos.

2. Ajusta parámetros:
   - **Seed**: para reproducibilidad.
   - **Block size**: tamaño de bloque (para métodos de bloque).
   - **Keep %**: porcentaje de datos a conservar (para métodos hybrid).
   - **Precio inicial**: precio de arranque para la reconstrucción.

3. Clic en **"Generar sintética"**.

**Qué verás:**
- Panel comparativo de 4 gráficos (precios, retornos, histograma, métricas).
- Métricas K-S p-value, ACF Retornos, ACF Retornos².
- Botón para descargar la sintética en CSV.

---

### Paso 4: Generación Masiva

**Pestaña: 🚀 Generación Masiva**

Genera **múltiples series sintéticas** que cumplan criterios de calidad estadística.

1. Configura:
   - **Nº sintéticas**: cuántas series válidas generar (ej: 5-20).
   - **Método bootstrap**: selecciona uno de los 5 métodos.
   - **Keep %** y **Block size**: parámetros del método.
   - **Umbral KS p-value**: mínimo para aceptar (ej: 0.95).
   - **Umbral ACF Ret**: mínimo correlación ACF retornos (ej: 0.80).
   - **Umbral ACF Ret²**: mínimo correlación ACF retornos² (ej: 0.80).
   - **Max iteraciones**: intentos máximos por sintética.

2. Clic en **"🚀 Generar múltiples"**.

3. Una vez generadas, selecciona cualquier sintética del dropdown para ver su panel comparativo.

4. Descarga individualmente con el botón de descarga, o **descarga todas en ZIP** (ver Paso 5).

> ⏱️ La generación puede tardar varios minutos según los umbrales y el tamaño del dataset.

---

### Paso 5: Ensemble

**Pestaña: 🧩 Ensemble**

Combina múltiples sintéticas en un solo dataset ensamblado.

**Métodos disponibles:**
- **mean**: promedio de precios OHLC.
- **median**: mediana de precios OHLC.
- **returns**: promedio de log-retornos → reconstrucción de precios.

**Funcionalidades:**
1. Construir ensemble y ver métricas de calidad.
2. Visualizar **bandas de confianza** (percentiles configurables).
3. **Votación de señales Target**: combina las predicciones Target de todas las sintéticas por votación con umbral configurable.
4. **Descargar ZIP** con todas las sintéticas + ensemble en un único archivo.

**Votación de señales:**
- Para cada observación, calcula qué proporción de sintéticas tienen Target > 0.
- Si la proporción supera el umbral (ej: 0.5), la señal combinada es 1.
- Útil para generar señales de trading más robustas.

---

### Paso 6: Test Adverso

**Pestaña: ⚔️ Test Adverso**

Evalúa la **robustez** de una estrategia/señal comparando su rendimiento en la serie original contra las sintéticas.

**Test Adverso por Señal:**
1. Define reglas de señal (formato pandas query).
2. La app aplica la señal a la serie original y a cada sintética.
3. Compara rendimientos: si el original está en el percentil ≥ 80%, la estrategia es robusta.

**Test Adverso Monkey:**
1. Genera N señales completamente aleatorias.
2. Calcula rendimientos en original y en sintéticas.
3. Si la correlación es alta, los datos sintéticos replican bien la dinámica original.

**Interpretación:**
- ✅ **ROBUSTA**: la estrategia funciona mejor en la serie real que en las sintéticas (no es sobreajuste).
- ❌ **NO ROBUSTA**: las sintéticas producen rendimientos similares o mejores (posible sobreajuste).

---

### Paso 7: Monkey Test

**Pestaña: 🐒 Monkey Test**

Simulación Out-of-Sample para evaluar la distribución de rendimientos posibles.

1. Configura el **nº de simulaciones** (ej: 1000).
2. Ajusta la **fracción de muestra** (0.1 a 1.0).
3. Selecciona el **quantil** de referencia (ej: 80%).
4. Clic en **"Ejecutar Monkey Test"**.

**Qué verás:**
- Histograma de retornos acumulados de todas las simulaciones.
- Líneas de media, mediana y quantil seleccionado.
- Estadísticas detalladas (media, mediana, std, min, max, quantil).

---

### Paso 8: Análisis Anual y Filtros

**Pestaña: 📅 Análisis Anual & Filtros**

1. **Rendimientos anuales**: tabla con el cambio % primer día → último día de cada año.
2. **Filtro por años positivos**: filtra el dataset para conservar solo años con rendimiento positivo.
3. **Evaluación de reglas**: escribe reglas tipo pandas query y evalúa su rendimiento en los datos filtrados.

**Ejemplo de reglas:**
```
close_relative > 0
open_relative < 0
high_relative > 0.005
```

---

## Estructura del Proyecto

```
synthetic_ohlc_project/
├── app.py                      # Interfaz Streamlit (punto de entrada)
├── run.sh                      # Script de arranque Linux/macOS
├── run.bat                     # Script de arranque Windows
├── requirements.txt            # Dependencias Python
├── README.md                   # Esta guía
├── core/
│   ├── __init__.py             # Exports del paquete
│   ├── ohlc_reference.py       # Parse OHLC ↔ vectores referenciados
│   ├── bootstrap_methods.py    # 5 métodos de bootstrap + generación masiva
│   ├── evaluation.py           # K-S test, ACF retornos, comparación visual
│   ├── data_loader.py          # Carga CSV, partición IS/OOS/Forward
│   ├── monkey_filters.py       # Monkey test, rendimientos anuales, filtros
│   ├── ensemble.py             # Ensemble: mean, median, returns, votación
│   └── adverse_test.py         # Test adverso de robustez
```

---

## Descripción de Módulos

| Módulo | Funciones Principales |
|--------|----------------------|
| `ohlc_reference.py` | `parse_ohlc_to_referenced`, `reconstruct_ohlc_from_referenced`, `get_summary_statistics`, `plot_referenced_vectors_histograms` |
| `bootstrap_methods.py` | `bootstrap_ohlc_simple`, `block_bootstrap_ohlc`, `intra_block_shuffle_bootstrap`, `hybrid_bootstrap_ohlc`, `block_aware_hybrid_bootstrap`, `generate_multiple_synthetics` |
| `evaluation.py` | `evaluate_synthetic_quality`, `compare_synthetic_data` |
| `data_loader.py` | `load_csv_data`, `split_dataset` |
| `monkey_filters.py` | `plot_histogram_simulacion_oos`, `rendimientos_primer_ultimo_dia`, `filtrar_años_positivos`, `evaluar_reglas_años_positivos`, `meta_regla_simple` |
| `ensemble.py` | `ensemble_mean`, `ensemble_median`, `ensemble_returns`, `ensemble_target_signals`, `plot_ensemble_comparison`, `compute_ensemble_bands`, `plot_ensemble_bands` |
| `adverse_test.py` | `adverse_test_returns`, `adverse_test_monkey` |

---

## Exportación de Datos

- **CSV individual**: cada sintética se puede descargar individualmente desde su pestaña.
- **ZIP completo**: desde la pestaña Ensemble, descarga un ZIP con:
  - Todas las sintéticas válidas (`synth_1.csv`, `synth_2.csv`, ...).
  - El ensemble resultante (`ensemble_{método}.csv`).
  - Las señales por votación (`ensemble_signals.csv`).

---

## Parámetros Recomendados

| Parámetro | Uso Típico | Rango Sugerido |
|-----------|-----------|----------------|
| Keep % | Porcentaje de datos originales a conservar | 0.3 – 0.7 |
| Block size | Tamaño de bloque para bootstrap | 3 – 20 |
| KS target | Umbral K-S p-value | 0.80 – 0.99 |
| ACF Ret target | Umbral ACF retornos | 0.70 – 0.90 |
| ACF Ret² target | Umbral ACF retornos² | 0.70 – 0.90 |
| Max iteraciones | Intentos por sintética | 500 – 5000 |
| Nº sintéticas | Cuántas series generar | 5 – 30 |
| Umbral votación | Consenso mínimo para señal | 0.4 – 0.7 |

> **Tip**: Umbrales muy altos (>0.95) pueden requerir muchas iteraciones. Empieza con umbrales moderados y ajusta.

---

## Solución de Problemas

| Problema | Solución |
|----------|----------|
| "No se encontró sintética en N intentos" | Reduce los umbrales KS/ACF o aumenta max iteraciones |
| "Error leyendo CSV" | Verifica que las columnas sean exactamente: Date, Open, High, Low, Close |
| "Vectores referenciados vacíos" | Asegúrate de tener al menos 2 filas de datos |
| La generación masiva es muy lenta | Reduce el nº de sintéticas o usa umbrales más bajos |
| "Módulo no encontrado" | Ejecuta `pip install -r requirements.txt` desde la carpeta del proyecto |
| Puerto ocupado | Usa `./run.sh --port 8502` para otro puerto |
