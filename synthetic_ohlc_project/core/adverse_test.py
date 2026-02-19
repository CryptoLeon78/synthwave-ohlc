"""
core/adverse_test.py
Test Adverso: evalúa si una estrategia/señal es robusta comparando su rendimiento
contra distribuciones generadas a partir de datos sintéticos.

Lógica: Si la estrategia funciona igual de bien (o mejor) en la serie original
que en las sintéticas, es señal de robustez. Si las sintéticas producen
rendimientos similares o superiores, la estrategia puede ser un artefacto
del sobreajuste.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Callable


def adverse_test_returns(
    original_data: pd.DataFrame,
    synthetic_datasets: Dict[str, pd.DataFrame],
    signal: pd.Series,
    target_col: str = "Target",
    price_col: str = "Close",
) -> Tuple[plt.Figure, Dict[str, float]]:
    """
    Test adverso basado en retornos: compara el rendimiento acumulado
    de una señal aplicada a la serie original vs aplicada a cada sintética.

    Args:
        original_data: DataFrame OHLC original con columna Target.
        synthetic_datasets: Diccionario {nombre: DataFrame OHLC sintético}.
        signal: Serie booleana/int con la señal de trading (1=operar, 0=no).
        target_col: Columna de rendimiento objetivo.
        price_col: Columna de precio.

    Returns:
        Tupla (Figure, dict con estadísticas del test).
    """
    # Rendimiento original con señal
    if target_col not in original_data.columns:
        raise ValueError(f"La columna '{target_col}' no existe en original_data.")

    signal_aligned = signal.reindex(original_data.index).fillna(0)
    orig_returns = original_data[target_col] * signal_aligned
    orig_cum = float((1 + orig_returns).prod() - 1)

    # Rendimiento en cada sintética
    synth_cums: List[float] = []
    for name, sdf in synthetic_datasets.items():
        if target_col in sdf.columns:
            sig = signal_aligned.reindex(sdf.index).fillna(0)
            s_ret = sdf[target_col] * sig
            synth_cums.append(float((1 + s_ret).prod() - 1))
        else:
            # Calcular Target para sintética si no existe
            sdf_target = (sdf["Open"].shift(-2) - sdf["Open"].shift(-1)) / sdf["Open"].shift(-1)
            sig = signal_aligned.reindex(sdf.index).fillna(0)
            s_ret = sdf_target.fillna(0) * sig
            synth_cums.append(float((1 + s_ret).prod() - 1))

    synth_arr = np.array(synth_cums)

    # ¿Cuántas sintéticas superan al original?
    n_better = int((synth_arr >= orig_cum).sum())
    percentile_rank = float((synth_arr < orig_cum).mean() * 100)

    stats = {
        "original_return": orig_cum,
        "synth_mean": float(np.mean(synth_arr)),
        "synth_median": float(np.median(synth_arr)),
        "synth_std": float(np.std(synth_arr)),
        "synth_min": float(np.min(synth_arr)) if len(synth_arr) > 0 else 0.0,
        "synth_max": float(np.max(synth_arr)) if len(synth_arr) > 0 else 0.0,
        "n_synthetics": len(synth_cums),
        "n_better_than_original": n_better,
        "percentile_rank": percentile_rank,
        "is_robust": percentile_rank >= 80,
    }

    # Gráfico
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histograma
    ax = axes[0]
    ax.hist(synth_arr, bins=30, alpha=0.7, edgecolor="black", color="#6baed6", label="Sintéticas")
    ax.axvline(orig_cum, color="red", linewidth=2.5, linestyle="--",
               label=f"Original: {orig_cum:.4f}")
    ax.axvline(np.mean(synth_arr), color="green", linewidth=1.5, linestyle="--",
               label=f"Media sint.: {np.mean(synth_arr):.4f}")
    ax.set_title("Test Adverso: Distribución de Rendimientos", fontsize=13, fontweight="bold")
    ax.set_xlabel("Retorno Acumulado")
    ax.set_ylabel("Frecuencia")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Resumen textual
    ax = axes[1]
    ax.axis("off")
    robust_txt = "✅ ROBUSTA" if stats["is_robust"] else "❌ NO ROBUSTA"
    text = (
        f"TEST ADVERSO — RESULTADO\n\n"
        f"Rendimiento original:  {orig_cum:.4f}\n"
        f"Media sintéticas:      {stats['synth_mean']:.4f}\n"
        f"Mediana sintéticas:    {stats['synth_median']:.4f}\n"
        f"Std sintéticas:        {stats['synth_std']:.4f}\n\n"
        f"Sintéticas que superan original: {n_better}/{len(synth_cums)}\n"
        f"Percentil del original: {percentile_rank:.1f}%\n\n"
        f"Veredicto: {robust_txt}\n"
        f"(Robusto si percentil ≥ 80%)"
    )
    ax.text(0.05, 0.5, text, fontsize=11, verticalalignment="center",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow"))

    fig.tight_layout()
    print(f"Test Adverso: original={orig_cum:.4f}, percentil={percentile_rank:.1f}%, "
          f"{n_better}/{len(synth_cums)} mejores → {robust_txt}")

    return fig, stats


def adverse_test_monkey(
    original_data: pd.DataFrame,
    synthetic_datasets: Dict[str, pd.DataFrame],
    target_col: str = "Target",
    n_random_signals: int = 500,
    signal_density: float = 0.5,
    seed: Optional[int] = None,
) -> Tuple[plt.Figure, Dict[str, float]]:
    """
    Test adverso tipo Monkey: genera señales aleatorias y compara rendimientos
    en la serie original vs en cada sintética.

    Si las señales aleatorias producen rendimientos similares en original y
    sintéticas, sugiere que no hay sobreajuste.

    Args:
        original_data: DataFrame con Target.
        synthetic_datasets: Dict de sintéticas.
        target_col: Columna objetivo.
        n_random_signals: Número de señales aleatorias a probar.
        signal_density: Proporción de 1s en cada señal aleatoria.
        seed: Semilla.

    Returns:
        Tupla (Figure, dict de stats).
    """
    if seed is not None:
        np.random.seed(seed)

    n = len(original_data)
    target_orig = original_data[target_col].fillna(0).values

    orig_rets: List[float] = []
    synth_rets_mean: List[float] = []

    for _ in range(n_random_signals):
        signal = (np.random.random(n) < signal_density).astype(float)
        o_ret = float((1 + target_orig * signal).prod() - 1)
        orig_rets.append(o_ret)

        s_rets = []
        for sdf in synthetic_datasets.values():
            if target_col in sdf.columns:
                t = sdf[target_col].fillna(0).values[:n]
            else:
                t = np.zeros(min(len(sdf), n))
            sig = signal[:len(t)]
            s_rets.append(float((1 + t * sig).prod() - 1))
        synth_rets_mean.append(float(np.mean(s_rets)))

    orig_arr = np.array(orig_rets)
    synth_arr = np.array(synth_rets_mean)

    corr = float(np.corrcoef(orig_arr, synth_arr)[0, 1]) if len(orig_arr) > 1 else 0.0

    stats = {
        "correlation_orig_synth": corr,
        "orig_mean": float(np.mean(orig_arr)),
        "synth_mean": float(np.mean(synth_arr)),
        "orig_std": float(np.std(orig_arr)),
        "synth_std": float(np.std(synth_arr)),
        "n_signals_tested": n_random_signals,
        "similar_behavior": abs(corr) > 0.5,
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.scatter(orig_arr, synth_arr, alpha=0.3, s=10, color="#e6550d")
    ax.set_xlabel("Retorno Original")
    ax.set_ylabel("Retorno Medio Sintéticas")
    ax.set_title(f"Monkey Adverso: corr={corr:.3f}", fontsize=13, fontweight="bold")
    ax.plot([min(orig_arr), max(orig_arr)], [min(orig_arr), max(orig_arr)],
            "k--", alpha=0.5, label="y=x")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.hist(orig_arr, bins=30, alpha=0.6, label="Original", color="blue")
    ax.hist(synth_arr, bins=30, alpha=0.6, label="Sint. (media)", color="orange")
    ax.set_title("Distribución de Retornos Aleatorios")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig, stats
