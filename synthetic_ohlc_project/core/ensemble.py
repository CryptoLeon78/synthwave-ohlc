"""
core/ensemble.py
Ensamblado (ensemble) de múltiples series sintéticas OHLC.
Replica la lógica de ensamblado del notebook "Sesión 4" (PyRE by Quantdemy).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple

from .evaluation import evaluate_synthetic_quality


# ---------------------------------------------------------------------------
# Ensemble por promedio de precios
# ---------------------------------------------------------------------------

def ensemble_mean(
    synthetic_datasets: Dict[str, pd.DataFrame],
    price_cols: List[str] = None,
) -> pd.DataFrame:
    """
    Crea un dataset ensamblado promediando los precios OHLC de todas
    las sintéticas proporcionadas.

    Args:
        synthetic_datasets: Diccionario {nombre: DataFrame OHLC}.
        price_cols: Columnas a promediar. Por defecto ['Open','High','Low','Close'].

    Returns:
        DataFrame OHLC ensamblado (promedio).
    """
    if not synthetic_datasets:
        raise ValueError("No hay datasets sintéticos para ensamblar.")

    if price_cols is None:
        price_cols = ["Open", "High", "Low", "Close"]

    dfs = list(synthetic_datasets.values())
    n = len(dfs)

    ensemble = dfs[0][price_cols].copy()
    for df in dfs[1:]:
        ensemble = ensemble + df[price_cols]
    ensemble = ensemble / n

    # Forzar coherencia OHLC
    ensemble["High"] = ensemble[["Open", "High", "Close"]].max(axis=1)
    ensemble["Low"] = ensemble[["Open", "Low", "Close"]].min(axis=1)

    print(f"Ensemble Mean: {n} sintéticas promediadas")
    return ensemble


def ensemble_median(
    synthetic_datasets: Dict[str, pd.DataFrame],
    price_cols: List[str] = None,
) -> pd.DataFrame:
    """
    Crea un dataset ensamblado usando la mediana de los precios OHLC.

    Args:
        synthetic_datasets: Diccionario {nombre: DataFrame OHLC}.
        price_cols: Columnas a calcular mediana.

    Returns:
        DataFrame OHLC ensamblado (mediana).
    """
    if not synthetic_datasets:
        raise ValueError("No hay datasets sintéticos para ensamblar.")

    if price_cols is None:
        price_cols = ["Open", "High", "Low", "Close"]

    stacked = {col: pd.concat(
        [df[col].rename(k) for k, df in synthetic_datasets.items()],
        axis=1,
    ) for col in price_cols}

    ensemble = pd.DataFrame({
        col: stacked[col].median(axis=1) for col in price_cols
    })

    ensemble["High"] = ensemble[["Open", "High", "Close"]].max(axis=1)
    ensemble["Low"] = ensemble[["Open", "Low", "Close"]].min(axis=1)

    print(f"Ensemble Median: {len(synthetic_datasets)} sintéticas")
    return ensemble


# ---------------------------------------------------------------------------
# Ensemble de retornos (promedio de log-retornos → reconstrucción)
# ---------------------------------------------------------------------------

def ensemble_returns(
    synthetic_datasets: Dict[str, pd.DataFrame],
    initial_price: float,
    price_col: str = "Close",
) -> pd.DataFrame:
    """
    Promedia los retornos logarítmicos de Close de todas las sintéticas
    y reconstruye una serie de precios a partir del precio inicial.

    Útil para ensamblar la dinámica de retornos en lugar de precios absolutos.

    Args:
        synthetic_datasets: Diccionario {nombre: DataFrame OHLC}.
        initial_price: Precio inicial para reconstrucción.
        price_col: Columna de precio.

    Returns:
        DataFrame con columna Close ensamblada por retornos.
    """
    if not synthetic_datasets:
        raise ValueError("No hay datasets sintéticos para ensamblar.")

    dfs = list(synthetic_datasets.values())
    n = len(dfs)

    # Calcular log-retornos de cada sintética
    all_returns = []
    for df in dfs:
        log_ret = np.log(df[price_col] / df[price_col].shift(1))
        all_returns.append(log_ret)

    returns_df = pd.concat(all_returns, axis=1)
    avg_returns = returns_df.mean(axis=1)
    avg_returns.iloc[0] = 0.0  # primer retorno = 0

    # Reconstruir precios
    cum_returns = avg_returns.cumsum()
    prices = initial_price * np.exp(cum_returns)

    result = pd.DataFrame({"Close": prices}, index=dfs[0].index)
    print(f"Ensemble Returns: {n} sintéticas → retornos promediados y reconstruidos")
    return result


# ---------------------------------------------------------------------------
# Ensemble de señales / Target
# ---------------------------------------------------------------------------

def ensemble_target_signals(
    synthetic_datasets: Dict[str, pd.DataFrame],
    target_col: str = "Target",
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Combina las señales Target de múltiples sintéticas mediante votación.

    Para cada observación, calcula la proporción de sintéticas con señal
    positiva y genera una señal final basada en el umbral de consenso.

    Args:
        synthetic_datasets: Diccionario {nombre: DataFrame OHLC con Target}.
        target_col: Nombre de la columna de señal.
        threshold: Proporción mínima de votos positivos para señal = 1.

    Returns:
        DataFrame con columnas: vote_ratio, ensemble_signal, avg_target.
    """
    if not synthetic_datasets:
        raise ValueError("No hay datasets sintéticos para ensamblar.")

    targets = pd.concat(
        [df[target_col].rename(k) for k, df in synthetic_datasets.items() if target_col in df.columns],
        axis=1,
    )

    if targets.empty:
        raise ValueError(f"Ninguna sintética contiene la columna '{target_col}'.")

    vote_ratio = (targets > 0).mean(axis=1)
    avg_target = targets.mean(axis=1)

    result = pd.DataFrame({
        "vote_ratio": vote_ratio,
        "ensemble_signal": (vote_ratio >= threshold).astype(int),
        "avg_target": avg_target,
    }, index=targets.index)

    n_positive = result["ensemble_signal"].sum()
    print(f"Ensemble Signals: {len(synthetic_datasets)} sintéticas, "
          f"umbral={threshold}, señales positivas={n_positive}/{len(result)}")
    return result


# ---------------------------------------------------------------------------
# Visualización del ensemble
# ---------------------------------------------------------------------------

def plot_ensemble_comparison(
    original_data: pd.DataFrame,
    synthetic_datasets: Dict[str, pd.DataFrame],
    ensemble_data: pd.DataFrame,
    price_col: str = "Close",
    nlags: int = 20,
) -> Tuple[plt.Figure, Dict[str, float]]:
    """
    Compara visualmente la serie original, las sintéticas individuales
    y el ensemble resultante.

    Subplot 1: Precios — original, sintéticas (gris) y ensemble (rojo).
    Subplot 2: Retornos del ensemble vs original.
    Subplot 3: Histogramas de retornos.
    Subplot 4: Métricas de calidad del ensemble.

    Args:
        original_data: DataFrame OHLC original.
        synthetic_datasets: Diccionario de sintéticas.
        ensemble_data: DataFrame OHLC del ensemble.
        price_col: Columna de precios.
        nlags: Lags para ACF.

    Returns:
        Tupla (Figure, dict con métricas del ensemble).
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 9))

    orig_prices = original_data[price_col].values
    ens_prices = ensemble_data[price_col].values

    orig_ret = np.log(original_data[price_col] / original_data[price_col].shift(1)).dropna()
    ens_ret = np.log(ensemble_data[price_col] / ensemble_data[price_col].shift(1)).dropna()

    # 1. Precios
    ax = axes[0, 0]
    for name, sdf in synthetic_datasets.items():
        ax.plot(sdf[price_col].values, color="gray", alpha=0.2, linewidth=0.5)
    ax.plot(orig_prices, label="Original", color="blue", alpha=0.9, linewidth=1.2)
    ax.plot(ens_prices, label="Ensemble", color="red", alpha=0.9, linewidth=1.5)
    ax.set_title(f"Precios: Original vs Ensemble ({len(synthetic_datasets)} sintéticas)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Retornos
    ax = axes[0, 1]
    ax.plot(ens_ret.values, label="Ensemble", alpha=0.8, color="red")
    ax.plot(orig_ret.values, label="Original", alpha=0.6, color="blue")
    ax.set_title("Retornos Logarítmicos")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Histograma
    ax = axes[1, 0]
    ax.hist(orig_ret, bins=50, alpha=0.6, label="Original", density=True, color="blue")
    ax.hist(ens_ret, bins=50, alpha=0.6, label="Ensemble", density=True, color="red")
    ax.set_title("Distribución de Retornos")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Métricas
    metrics = evaluate_synthetic_quality(orig_ret.values, ens_ret.values, nlags=nlags)

    ax = axes[1, 1]
    ax.axis("off")
    ks_s = "PASS" if metrics["ks_pvalue"] > 0.05 else "FAIL"
    ret_s = "PASS" if metrics["ret_correlation"] > 0.8 else "FAIL"
    sq_s = "PASS" if metrics["sq_correlation"] > 0.8 else "FAIL"

    text = (
        f"MÉTRICAS DEL ENSEMBLE\n"
        f"({len(synthetic_datasets)} sintéticas combinadas)\n\n"
        f"K-S p-value: {metrics['ks_pvalue']:.4f} [{ks_s}]\n"
        f"Distribuciones similares\n\n"
        f"ACF Retornos: {metrics['ret_correlation']:.4f} [{ret_s}]\n"
        f"Dependencia temporal\n\n"
        f"ACF Retornos²: {metrics['sq_correlation']:.4f} [{sq_s}]\n"
        f"Clustering volatilidad"
    )
    ax.text(
        0.1, 0.5, text, fontsize=11, verticalalignment="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"),
    )

    fig.tight_layout()
    return fig, {
        "ks_pvalue": metrics["ks_pvalue"],
        "ret_correlation": metrics["ret_correlation"],
        "sq_correlation": metrics["sq_correlation"],
    }


# ---------------------------------------------------------------------------
# Bandas de confianza del ensemble
# ---------------------------------------------------------------------------

def compute_ensemble_bands(
    synthetic_datasets: Dict[str, pd.DataFrame],
    price_col: str = "Close",
    lower_pct: float = 5.0,
    upper_pct: float = 95.0,
) -> pd.DataFrame:
    """
    Calcula bandas de confianza (percentiles) a partir de múltiples sintéticas.

    Útil para visualizar la dispersión y construir intervalos de confianza
    sobre la evolución de precios.

    Args:
        synthetic_datasets: Diccionario de sintéticas.
        price_col: Columna de precio.
        lower_pct: Percentil inferior (default 5).
        upper_pct: Percentil superior (default 95).

    Returns:
        DataFrame con mean, median, lower, upper.
    """
    prices = pd.concat(
        [df[price_col].rename(k) for k, df in synthetic_datasets.items()],
        axis=1,
    )

    bands = pd.DataFrame({
        "mean": prices.mean(axis=1),
        "median": prices.median(axis=1),
        "lower": prices.quantile(lower_pct / 100, axis=1),
        "upper": prices.quantile(upper_pct / 100, axis=1),
    }, index=prices.index)

    print(f"Bandas de confianza: [{lower_pct}%, {upper_pct}%] sobre {len(synthetic_datasets)} sintéticas")
    return bands


def plot_ensemble_bands(
    original_data: pd.DataFrame,
    bands: pd.DataFrame,
    price_col: str = "Close",
) -> plt.Figure:
    """
    Grafica la serie original con bandas de confianza del ensemble.

    Args:
        original_data: DataFrame OHLC original.
        bands: DataFrame con mean, median, lower, upper.
        price_col: Columna de precio original.

    Returns:
        Figure de matplotlib.
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.fill_between(
        range(len(bands)), bands["lower"], bands["upper"],
        alpha=0.2, color="orange", label=f"Banda de confianza",
    )
    ax.plot(bands["mean"].values, color="red", linewidth=1.2, label="Ensemble media")
    ax.plot(bands["median"].values, color="darkred", linewidth=0.8, linestyle="--", label="Ensemble mediana")
    ax.plot(original_data[price_col].values[:len(bands)], color="blue", linewidth=1.2, label="Original")

    ax.set_title("Serie Original con Bandas de Confianza del Ensemble")
    ax.set_xlabel("Observaciones")
    ax.set_ylabel("Precio")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig
