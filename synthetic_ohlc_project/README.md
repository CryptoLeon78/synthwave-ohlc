# SyntheticData QXPRO vQDemy

Generador de data sintética OHLC con métodos de bootstrap y evaluación estadística.  
Replica fielmente la lógica de **"Sesión 4: Data Sintética, Ensamblado y Filtros"** (PyRE by Quantdemy).

## Estructura

```
synthetic_ohlc_project/
├── app.py                      # Streamlit UI
├── requirements.txt
├── core/
│   ├── __init__.py
│   ├── ohlc_reference.py       # Parse OHLC ↔ referenced vectors
│   ├── bootstrap_methods.py    # 5 métodos de bootstrap + generación masiva
│   ├── evaluation.py           # K-S test, ACF retornos, comparación visual
│   ├── data_loader.py          # Carga CSV, partición IS/OOS/Forward
│   └── monkey_filters.py       # Monkey test, rendimientos anuales, filtros
```

## Instalación

```bash
cd synthetic_ohlc_project
pip install -r requirements.txt
```

## Ejecución

```bash
streamlit run app.py
```

## Funcionalidades

1. **Carga CSV** con columnas Date, Open, High, Low, Close
2. **Conversión a OHLC referenciado** al close anterior
3. **5 métodos de bootstrap**: Simple, Block, Intra-Block Shuffle, Hybrid, Block-Aware Hybrid
4. **Evaluación de calidad**: K-S p-value, ACF retornos, ACF retornos²
5. **Generación masiva** con umbrales configurables
6. **Monkey Test** (simulación OOS)
7. **Análisis anual** y filtrado por años positivos
8. **Exportación CSV** de series sintéticas válidas

## CSV esperado

```csv
Date,Open,High,Low,Close
2010-01-04,1.4320,1.4440,1.4280,1.4410
...
```
