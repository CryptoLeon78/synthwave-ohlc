"""
core/evaluation.py
Evaluación de calidad de data sintética: K-S test, ACF retornos, ACF retornos².
Comparación visual original vs sintética.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional
from scipy import stats
from statsmodels.tsa.stattools import acf


def evaluate_synthetic_quality(
    original_returns: np.ndarray,
    synthetic_returns: np.ndarray,
    nlags: int = 60,
) -> Dict[str, float]:
    """
    Evalúa la calidad de data sintética según 3 criterios de Alan Tomillero:

    1. Kolmogorov-Smirnov p-value entre distribuciones de retornos.
    2. Correlación entre ACF de retornos (original vs sintético).
    3. Correlación entre ACF de retornos al cuadrado.

    Args:
        original_returns: Retornos logarítmicos de la serie original.
        synthetic_returns: Retornos logarítmicos de la serie sintética.
        nlags: Número de lags para el ACF.

    Returns:
        Diccionario con ks_pvalue, ret_correlation, sq_correlation, meets_criteria.
    """
    # 1. Kolmogorov-Smirnov
    _, p_value = stats.ks_2samp(original_returns, synthetic_returns)

    # 2. ACF de retornos
    acf_orig = acf(original_returns, nlags=nlags, fft=True)
    acf_synt = acf(synthetic_returns, nlags=nlags, fft=True)
    ret_corr = float(np.corrcoef(acf_orig[1:], acf_synt[1:])[0, 1])

    # 3. ACF de retornos²
    acf_orig_sq = acf(original_returns ** 2, nlags=nlags, fft=True)
    acf_synt_sq = acf(synthetic_returns ** 2, nlags=nlags, fft=True)
    sq_corr = float(np.corrcoef(acf_orig_sq[1:], acf_synt_sq[1:])[0, 1])

    return {
        "ks_pvalue": float(p_value),
        "ret_correlation": ret_corr,
        "sq_correlation": sq_corr,
        "meets_criteria": (p_value > 0.8 and ret_corr > 0.8 and sq_corr > 0.8),
    }


def compare_synthetic_data(
    original_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    price_col: str = "Close",
    nlags: int = 20,
) -> Tuple[plt.Figure, Dict[str, float]]:
    """
    Compara data original vs sintética con 4 subplots y métricas.

    Subplot 1: Precios Close.
    Subplot 2: Retornos logarítmicos.
    Subplot 3: Histogramas de retornos.
    Subplot 4: Métricas K-S, ACF Ret, ACF Ret² con PASS/FAIL.

    Args:
        original_data: DataFrame OHLC original.
        synthetic_data: DataFrame OHLC sintético.
        price_col: Columna de precios a comparar.
        nlags: Lags para ACF.

    Returns:
        Tupla (Figure, dict con métricas).
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 8))

    # Retornos
    orig_ret = np.log(original_data[price_col] / original_data[price_col].shift(1)).dropna()
    synt_ret = np.log(synthetic_data[price_col] / synthetic_data[price_col].shift(1)).dropna()

    # 1. Precios
    ax = axes[0, 0]
    ax.plot(synthetic_data[price_col].values, label="Sintético", alpha=0.8)
    ax.plot(original_data[price_col].iloc[1:].values, label="Original", alpha=0.8)
    ax.set_title("Comparación de Precios")
    ax.set_ylabel("Precio")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Retornos
    ax = axes[0, 1]
    ax.plot(synt_ret.values, label="Sintético", alpha=0.8)
    ax.plot(orig_ret.values, label="Original", alpha=0.8)
    ax.set_title("Comparación de Retornos")
    ax.set_ylabel("Retorno")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Histograma
    ax = axes[1, 0]
    ax.hist(orig_ret, bins=50, alpha=0.6, label="Original", density=True)
    ax.hist(synt_ret, bins=50, alpha=0.6, label="Sintético", density=True)
    ax.set_title("Distribución de Retornos")
    ax.set_xlabel("Retorno")
    ax.set_ylabel("Densidad")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Métricas
    metrics = evaluate_synthetic_quality(orig_ret.values, synt_ret.values, nlags=nlags)

    ax = axes[1, 1]
    ax.axis("off")
    ks_s = "PASS" if metrics["ks_pvalue"] > 0.05 else "FAIL"
    ret_s = "PASS" if metrics["ret_correlation"] > 0.8 else "FAIL"
    sq_s = "PASS" if metrics["sq_correlation"] > 0.8 else "FAIL"

    text = (
        f"MÉTRICAS DE CALIDAD\n\n"
        f"K-S p-value: {metrics['ks_pvalue']:.4f} [{ks_s}]\n"
        f"Distribuciones similares\n\n"
        f"ACF Retornos: {metrics['ret_correlation']:.4f} [{ret_s}]\n"
        f"Dependencia temporal\n\n"
        f"ACF Retornos²: {metrics['sq_correlation']:.4f} [{sq_s}]\n"
        f"Clustering volatilidad"
    )
    ax.text(
        0.1, 0.5, text, fontsize=11, verticalalignment="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"),
    )

    fig.tight_layout()
    return fig, {
        "ks_pvalue": metrics["ks_pvalue"],
        "ret_correlation": metrics["ret_correlation"],
        "sq_correlation": metrics["sq_correlation"],
    }
