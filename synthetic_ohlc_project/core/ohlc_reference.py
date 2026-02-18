"""
core/ohlc_reference.py
Conversión OHLC ↔ vectores referenciados al close anterior, validación y estadísticas.
Replica fielmente la lógica de "Sesión 4: Data Sintética" (PyRE by Quantdemy).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Validación estructural
# ---------------------------------------------------------------------------

def _validate_ohlc_structure(ohlc_data: pd.DataFrame) -> None:
    """
    Valida que los datos OHLC cumplan las restricciones estructurales básicas.

    Comprobaciones:
        - High >= max(Open, Close)
        - Low  <= min(Open, Close)
        - High >= Low
        - Precios estrictamente positivos

    Args:
        ohlc_data: DataFrame con columnas ['Open', 'High', 'Low', 'Close'].

    Raises:
        ValueError: Si hay violaciones estructurales.
    """
    violations: list[str] = []

    high_violation = ~(ohlc_data["High"] >= np.maximum(ohlc_data["Open"], ohlc_data["Close"]))
    if high_violation.any():
        violations.append(f"High < max(Open, Close) en {high_violation.sum()} observaciones")

    low_violation = ~(ohlc_data["Low"] <= np.minimum(ohlc_data["Open"], ohlc_data["Close"]))
    if low_violation.any():
        violations.append(f"Low > min(Open, Close) en {low_violation.sum()} observaciones")

    range_violation = ~(ohlc_data["High"] >= ohlc_data["Low"])
    if range_violation.any():
        violations.append(f"High < Low en {range_violation.sum()} observaciones")

    negative_prices = (ohlc_data[["Open", "High", "Low", "Close"]] <= 0).any(axis=1)
    if negative_prices.any():
        violations.append(f"Precios <= 0 en {negative_prices.sum()} observaciones")

    if violations:
        msg = "Violaciones estructurales OHLC encontradas:\n" + "\n".join(f"- {v}" for v in violations)
        raise ValueError(msg)


# ---------------------------------------------------------------------------
# OHLC → Vectores referenciados
# ---------------------------------------------------------------------------

def parse_ohlc_to_referenced(
    ohlc_data: pd.DataFrame,
    validate_input: bool = True,
    handle_missing: str = "drop",
) -> pd.DataFrame:
    """
    Convierte datos OHLC a representación referenciada al close anterior.

    Cada vela se transforma en un vector de 4 componentes que expresan los
    precios como cambios porcentuales respecto al Close_{t-1}.

    Args:
        ohlc_data: DataFrame con columnas ['Open', 'High', 'Low', 'Close'].
        validate_input: Si True, valida coherencia estructural del OHLC.
        handle_missing: Estrategia para valores faltantes —
            'drop', 'interpolate' o 'fill_forward'.

    Returns:
        DataFrame con columnas:
            open_relative, high_relative, low_relative, close_relative.

    Raises:
        ValueError: Columnas faltantes o datos insuficientes.
    """
    required = ["Open", "High", "Low", "Close"]
    missing = [c for c in required if c not in ohlc_data.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")
    if len(ohlc_data) < 2:
        raise ValueError("Se requieren al menos 2 observaciones")

    data = ohlc_data[required].copy()

    # Manejo de missing values
    if handle_missing == "drop":
        data = data.dropna()
    elif handle_missing == "interpolate":
        data = data.interpolate(method="linear")
    elif handle_missing == "fill_forward":
        data = data.ffill().dropna()
    else:
        raise ValueError("handle_missing debe ser 'drop', 'interpolate' o 'fill_forward'")

    if len(data) < 2:
        raise ValueError("Después de manejar valores faltantes, no quedan suficientes observaciones")

    if validate_input:
        _validate_ohlc_structure(data)

    prev_close = data["Close"].shift(1)

    referenced_vectors = pd.DataFrame(
        {
            "open_relative": (data["Open"] - prev_close) / prev_close,
            "high_relative": (data["High"] - prev_close) / prev_close,
            "low_relative": (data["Low"] - prev_close) / prev_close,
            "close_relative": (data["Close"] - prev_close) / prev_close,
        },
        index=data.index,
    )

    # Primera fila sin close anterior
    referenced_vectors = referenced_vectors.iloc[1:].copy()

    if referenced_vectors.isnull().any().any():
        print("⚠️  Advertencia: Se encontraron valores NaN en los vectores referenciados")

    if np.isinf(referenced_vectors.values).any():
        print("⚠️  Advertencia: Se encontraron valores infinitos en los vectores referenciados")
        referenced_vectors = referenced_vectors.replace([np.inf, -np.inf], np.nan).dropna()

    print(f"✓ Parseo completado: {len(data)} → {len(referenced_vectors)} vectores referenciados")
    print("✓ Primera observación perdida (no tiene close anterior)")

    return referenced_vectors


# ---------------------------------------------------------------------------
# Vectores referenciados → OHLC
# ---------------------------------------------------------------------------

def reconstruct_ohlc_from_referenced(
    referenced_vectors: pd.DataFrame,
    initial_price: float = 100.0,
    validate_output: bool = True,
) -> pd.DataFrame:
    """
    Reconstruye datos OHLC a partir de vectores referenciados al close anterior.

    Args:
        referenced_vectors: DataFrame con columnas referenciadas.
        initial_price: Precio Close_{0} para arrancar la reconstrucción.
        validate_output: Si True, valida la coherencia del OHLC resultante.

    Returns:
        DataFrame con columnas ['Open', 'High', 'Low', 'Close'] reconstruidas.
    """
    required = ["open_relative", "high_relative", "low_relative", "close_relative"]
    missing = [c for c in required if c not in referenced_vectors.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")
    if len(referenced_vectors) == 0:
        raise ValueError("No hay vectores para reconstruir")

    rows = []
    current_close = initial_price

    for _, row in referenced_vectors.iterrows():
        new_open = current_close * (1 + row["open_relative"])
        new_high = current_close * (1 + row["high_relative"])
        new_low = current_close * (1 + row["low_relative"])
        new_close = current_close * (1 + row["close_relative"])

        # Forzar coherencia estructural
        new_high = max(new_high, new_open, new_close)
        new_low = min(new_low, new_open, new_close)

        rows.append({"Open": new_open, "High": new_high, "Low": new_low, "Close": new_close})
        current_close = new_close

    result = pd.DataFrame(rows, index=referenced_vectors.index)

    if validate_output:
        try:
            _validate_ohlc_structure(result)
        except ValueError as e:
            print(f"⚠️  Advertencia en OHLC reconstruido: {e}")

    return result


# ---------------------------------------------------------------------------
# Estadísticas descriptivas y visualización
# ---------------------------------------------------------------------------

def get_summary_statistics(referenced_vectors: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula estadísticas descriptivas extendidas de los vectores referenciados.

    Incluye describe() + skew, kurtosis y una "special_condition":
        - open_relative : proporción de ceros
        - high_relative : proporción de negativos
        - low_relative  : proporción de positivos
        - close_relative: proporción de positivos

    Returns:
        DataFrame con las estadísticas.
    """
    desc = referenced_vectors.describe()

    additional = pd.DataFrame(
        {
            "open_relative": [
                referenced_vectors["open_relative"].skew(),
                referenced_vectors["open_relative"].kurtosis(),
                (referenced_vectors["open_relative"] == 0).mean(),
            ],
            "high_relative": [
                referenced_vectors["high_relative"].skew(),
                referenced_vectors["high_relative"].kurtosis(),
                (referenced_vectors["high_relative"] < 0).mean(),
            ],
            "low_relative": [
                referenced_vectors["low_relative"].skew(),
                referenced_vectors["low_relative"].kurtosis(),
                (referenced_vectors["low_relative"] > 0).mean(),
            ],
            "close_relative": [
                referenced_vectors["close_relative"].skew(),
                referenced_vectors["close_relative"].kurtosis(),
                (referenced_vectors["close_relative"] > 0).mean(),
            ],
        },
        index=["skew", "kurtosis", "special_condition"],
    )

    return pd.concat([desc, additional])


def plot_referenced_vectors_histograms(
    referenced_vectors: pd.DataFrame,
    bins: int = 50,
    figsize: Tuple[int, int] = (20, 5),
) -> plt.Figure:
    """
    Histogramas de los 4 componentes del vector OHLC referenciado.

    Args:
        referenced_vectors: Vectores OHLC referenciados.
        bins: Número de bins.
        figsize: Tamaño de la figura.

    Returns:
        Objeto Figure de matplotlib.
    """
    fig, axes = plt.subplots(1, 4, figsize=figsize)

    components = ["open_relative", "high_relative", "low_relative", "close_relative"]
    titles = ["Open Relative", "High Relative", "Low Relative", "Close Relative"]
    colors = ["#1f77b4", "#2ca02c", "#d62728", "#ff7f0e"]

    for i, (comp, title, color) in enumerate(zip(components, titles, colors)):
        if comp in referenced_vectors.columns:
            axes[i].hist(referenced_vectors[comp], bins=bins, alpha=0.7, color=color, edgecolor="black")
            axes[i].set_title(title, fontsize=12, fontweight="bold")
            axes[i].set_xlabel("Valor Relativo")
            axes[i].set_ylabel("Frecuencia")
            axes[i].grid(True, alpha=0.3)
            axes[i].axvline(x=0, color="black", linestyle="--", alpha=0.5)
            mean_val = referenced_vectors[comp].mean()
            axes[i].axvline(x=mean_val, color="red", linestyle="-", alpha=0.8, linewidth=2, label=f"Media: {mean_val:.6f}")
            axes[i].legend(fontsize=8)

    fig.suptitle("Distribuciones de Vectores OHLC Referenciados al Close Anterior", fontsize=14, y=1.02)
    fig.tight_layout()
    return fig
