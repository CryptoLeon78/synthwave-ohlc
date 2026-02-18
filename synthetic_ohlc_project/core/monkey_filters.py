"""
core/monkey_filters.py
Monkey test (simulación OOS), rendimientos anuales, filtrado por años positivos,
evaluación de reglas y meta-regla.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional


# ---------------------------------------------------------------------------
# Monkey Test / Simulación OOS
# ---------------------------------------------------------------------------

def plot_histogram_simulacion_oos(
    df_oos: pd.DataFrame,
    target_col: str = "Target",
    n_simulations: int = 1000,
    sample_fraction: float = 0.5,
    quantile: int = 80,
    seed: Optional[int] = None,
    figsize: Tuple[int, int] = (12, 6),
) -> Tuple[plt.Figure, Dict[str, float]]:
    """
    Realiza simulaciones tomando fracciones aleatorias del OOS y calcula
    el retorno acumulado de cada simulación. Dibuja histograma con estadísticas.

    Args:
        df_oos: DataFrame con columna Target.
        target_col: Nombre de la columna objetivo.
        n_simulations: Número de simulaciones.
        sample_fraction: Fracción del dataset a tomar en cada simulación.
        quantile: Percentil a mostrar (por defecto 80).
        seed: Semilla para reproducibilidad.
        figsize: Tamaño de la figura.

    Returns:
        Tupla (Figure, dict con media, mediana, std, quantil, min, max).
    """
    if seed is not None:
        np.random.seed(seed)

    target = df_oos[target_col].dropna()
    n_sample = max(1, int(len(target) * sample_fraction))

    cum_returns: List[float] = []
    for _ in range(n_simulations):
        sample = target.sample(n=n_sample, replace=True)
        cum_ret = (1 + sample).prod() - 1
        cum_returns.append(cum_ret)

    cum_returns_arr = np.array(cum_returns)

    stats_dict = {
        "media": float(np.mean(cum_returns_arr)),
        "mediana": float(np.median(cum_returns_arr)),
        "std": float(np.std(cum_returns_arr)),
        f"quantil_{quantile}": float(np.percentile(cum_returns_arr, quantile)),
        "min": float(np.min(cum_returns_arr)),
        "max": float(np.max(cum_returns_arr)),
    }

    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(cum_returns_arr, bins=50, alpha=0.7, edgecolor="black", color="#4a90d9")
    ax.axvline(stats_dict["media"], color="red", linestyle="--", linewidth=2, label=f"Media: {stats_dict['media']:.4f}")
    ax.axvline(stats_dict["mediana"], color="green", linestyle="--", linewidth=2, label=f"Mediana: {stats_dict['mediana']:.4f}")
    ax.axvline(stats_dict[f"quantil_{quantile}"], color="orange", linestyle="--", linewidth=2,
               label=f"Q{quantile}: {stats_dict[f'quantil_{quantile}']:.4f}")
    ax.set_title(f"Monkey Test: {n_simulations} simulaciones OOS", fontsize=14, fontweight="bold")
    ax.set_xlabel("Retorno Acumulado")
    ax.set_ylabel("Frecuencia")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    return fig, stats_dict


# ---------------------------------------------------------------------------
# Rendimientos anuales primer–último día
# ---------------------------------------------------------------------------

def rendimientos_primer_ultimo_dia(
    df: pd.DataFrame,
    price_col: str = "Close",
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Agrupa por año y calcula el cambio porcentual del primer al último día.

    Args:
        df: DataFrame con índice datetime y columna de precio.
        price_col: Columna de precio a usar.

    Returns:
        Tupla (DataFrame con rendimientos anuales, dict con estadísticas).
    """
    annual = df.groupby(df.index.year)[price_col].agg(["first", "last"])
    annual["return_pct"] = (annual["last"] - annual["first"]) / annual["first"] * 100
    annual.index.name = "year"

    stats_dict = {
        "media": float(annual["return_pct"].mean()),
        "std": float(annual["return_pct"].std()),
        "mejor_año": int(annual["return_pct"].idxmax()),
        "peor_año": int(annual["return_pct"].idxmin()),
        "pct_positivos": float((annual["return_pct"] > 0).mean() * 100),
    }

    print(f"Rendimientos anuales — Media: {stats_dict['media']:.2f}%, "
          f"Std: {stats_dict['std']:.2f}%, "
          f"Años positivos: {stats_dict['pct_positivos']:.1f}%")

    return annual, stats_dict


# ---------------------------------------------------------------------------
# Filtrado de años positivos
# ---------------------------------------------------------------------------

def filtrar_años_positivos(
    df: pd.DataFrame,
    annual_returns: pd.DataFrame,
    price_col: str = "Close",
) -> Tuple[pd.DataFrame, List[int]]:
    """
    Filtra el DataFrame para conservar solo los años con rendimiento positivo.

    Args:
        df: DataFrame original completo.
        annual_returns: DataFrame de rendimientos anuales (de rendimientos_primer_ultimo_dia).
        price_col: Columna de precio.

    Returns:
        Tupla (DataFrame filtrado, lista de años positivos).
    """
    positive_years = annual_returns[annual_returns["return_pct"] > 0].index.tolist()
    df_positive = df[df.index.year.isin(positive_years)].copy()

    # Calcular rendimiento total
    total_all = (df[price_col].iloc[-1] / df[price_col].iloc[0] - 1) * 100
    total_pos = (df_positive[price_col].iloc[-1] / df_positive[price_col].iloc[0] - 1) * 100 if len(df_positive) > 0 else 0

    print(f"Años positivos: {positive_years}")
    print(f"Rendimiento total (todos): {total_all:.2f}%")
    print(f"Rendimiento total (solo positivos): {total_pos:.2f}%")

    return df_positive, positive_years


# ---------------------------------------------------------------------------
# Evaluación de reglas en años positivos
# ---------------------------------------------------------------------------

def evaluar_reglas_años_positivos(
    train_positivos: pd.DataFrame,
    rules: List[str],
    target_col: str = "Target",
) -> pd.DataFrame:
    """
    Evalúa una lista de reglas (expresiones query de pandas) sobre un
    DataFrame de años positivos. Calcula el rendimiento de cada regla.

    Args:
        train_positivos: DataFrame con solo años positivos y columna Target.
        rules: Lista de strings compatibles con DataFrame.query().
        target_col: Columna objetivo.

    Returns:
        DataFrame con regla, rendimiento, ordenado descendente.
    """
    results = []
    for rule in rules:
        try:
            filtered = train_positivos.query(rule)
            if len(filtered) > 0:
                ret = (1 + filtered[target_col]).prod() - 1
            else:
                ret = 0.0
            results.append({"rule": rule, "return": ret, "n_signals": len(filtered)})
        except Exception as e:
            results.append({"rule": rule, "return": np.nan, "n_signals": 0, "error": str(e)})

    result_df = pd.DataFrame(results).sort_values("return", ascending=False)
    return result_df


# ---------------------------------------------------------------------------
# Meta-regla simple
# ---------------------------------------------------------------------------

def meta_regla_simple(
    df: pd.DataFrame,
    rules: List[str],
    target_col: str = "Target",
) -> Tuple[pd.Series, pd.Series]:
    """
    Combina múltiples reglas con OR: si cualquier regla se activa, se
    genera señal. Rendimiento = Target * señal * -1.

    Args:
        df: DataFrame con columna Target.
        rules: Lista de expresiones query.
        target_col: Columna objetivo.

    Returns:
        Tupla (señales combinadas, rendimientos).
    """
    combined_signal = pd.Series(False, index=df.index)

    for rule in rules:
        try:
            mask = df.eval(rule)
            combined_signal = combined_signal | mask
        except Exception as e:
            print(f"⚠️ Error en regla '{rule}': {e}")

    signal_int = combined_signal.astype(int)
    rendimientos = df[target_col] * signal_int * -1

    print(f"Meta-regla: {combined_signal.sum()}/{len(df)} señales activas ({combined_signal.mean()*100:.1f}%)")
    print(f"Rendimiento acumulado: {(1 + rendimientos).prod() - 1:.4f}")

    return signal_int, rendimientos
