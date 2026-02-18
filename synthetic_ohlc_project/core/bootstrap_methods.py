"""
core/bootstrap_methods.py
Métodos de bootstrap para generar data sintética OHLC.
Replica fielmente los 5 métodos del notebook "Sesión 4" (PyRE by Quantdemy).
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from tqdm import tqdm

from .ohlc_reference import reconstruct_ohlc_from_referenced
from .evaluation import evaluate_synthetic_quality


# ---------------------------------------------------------------------------
# 1. Bootstrap Simple
# ---------------------------------------------------------------------------

def bootstrap_ohlc_simple(
    referenced_vectors: pd.DataFrame,
    initial_price: float,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Re-muestreo con reemplazo sobre las filas de referenced_vectors.

    Args:
        referenced_vectors: Vectores OHLC referenciados.
        initial_price: Precio inicial para reconstrucción.
        seed: Semilla para reproducibilidad.

    Returns:
        DataFrame OHLC sintético reconstruido.
    """
    if seed is not None:
        np.random.seed(seed)

    n = len(referenced_vectors)
    indices = np.random.choice(n, size=n, replace=True)
    bootstrapped = referenced_vectors.iloc[indices].copy()
    bootstrapped.index = referenced_vectors.index

    return reconstruct_ohlc_from_referenced(bootstrapped, initial_price, validate_output=False)


# ---------------------------------------------------------------------------
# 2. Block Bootstrap
# ---------------------------------------------------------------------------

def block_bootstrap_ohlc(
    referenced_vectors: pd.DataFrame,
    initial_price: float,
    block_size: int = 5,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Selecciona bloques consecutivos de tamaño block_size, los concatena
    aleatoriamente hasta alcanzar el tamaño original.

    Preserva dependencias temporales locales.

    Args:
        referenced_vectors: Vectores OHLC referenciados.
        initial_price: Precio inicial para reconstrucción.
        block_size: Tamaño de cada bloque consecutivo.
        seed: Semilla para reproducibilidad.

    Returns:
        DataFrame OHLC sintético reconstruido.
    """
    if seed is not None:
        np.random.seed(seed)

    n = len(referenced_vectors)
    n_blocks_needed = int(np.ceil(n / block_size))
    max_start = n - block_size

    if max_start < 0:
        raise ValueError(f"block_size ({block_size}) mayor que la serie ({n})")

    blocks = []
    for _ in range(n_blocks_needed):
        start = np.random.randint(0, max_start + 1)
        block = referenced_vectors.iloc[start : start + block_size]
        blocks.append(block)

    bootstrapped = pd.concat(blocks, ignore_index=True).iloc[:n].copy()
    bootstrapped.index = referenced_vectors.index

    print(f"Block Bootstrap: {n_blocks_needed} bloques × {block_size} → {n} observaciones")

    return reconstruct_ohlc_from_referenced(bootstrapped, initial_price, validate_output=False)


# ---------------------------------------------------------------------------
# 3. Intra-Block Shuffle Bootstrap
# ---------------------------------------------------------------------------

def intra_block_shuffle_bootstrap(
    referenced_vectors: pd.DataFrame,
    initial_price: float,
    block_size: int = 5,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Divide la serie en bloques consecutivos de tamaño block_size.
    Dentro de cada bloque baraja las observaciones, pero el bloque
    permanece en la misma posición temporal.

    Args:
        referenced_vectors: Vectores OHLC referenciados.
        initial_price: Precio inicial para reconstrucción.
        block_size: Tamaño de cada bloque.
        seed: Semilla para reproducibilidad.

    Returns:
        DataFrame OHLC sintético reconstruido.
    """
    if seed is not None:
        np.random.seed(seed)

    n = len(referenced_vectors)
    shuffled_parts = []

    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        block = referenced_vectors.iloc[start:end].copy()
        shuffled_idx = np.random.permutation(len(block))
        block = block.iloc[shuffled_idx]
        shuffled_parts.append(block)

    bootstrapped = pd.concat(shuffled_parts, ignore_index=True)
    bootstrapped.index = referenced_vectors.index

    n_blocks = int(np.ceil(n / block_size))
    print(f"Intra-Block Shuffle: {n_blocks} bloques, shuffle interno en cada uno")

    return reconstruct_ohlc_from_referenced(bootstrapped, initial_price, validate_output=False)


# ---------------------------------------------------------------------------
# 4. Hybrid Bootstrap
# ---------------------------------------------------------------------------

def hybrid_bootstrap_ohlc(
    referenced_vectors: pd.DataFrame,
    initial_price: float,
    keep_percentage: float = 0.5,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Conserva un porcentaje de observaciones originales en su posición y
    reemplaza el resto por vectores aleatorios de la serie.

    Args:
        referenced_vectors: Vectores OHLC referenciados.
        initial_price: Precio inicial para reconstrucción.
        keep_percentage: Fracción de observaciones a conservar (0‑1).
        seed: Semilla para reproducibilidad.

    Returns:
        DataFrame OHLC sintético reconstruido.
    """
    if seed is not None:
        np.random.seed(seed)

    n = len(referenced_vectors)
    n_keep = int(n * keep_percentage)
    n_replace = n - n_keep

    # Índices a conservar (aleatorios)
    keep_indices = np.sort(np.random.choice(n, size=n_keep, replace=False))
    replace_indices = np.array([i for i in range(n) if i not in keep_indices])

    bootstrapped = referenced_vectors.copy()

    # Reemplazar con vectores aleatorios (sin reusar el propio índice)
    for idx in replace_indices:
        candidates = [i for i in range(n) if i != idx]
        donor = np.random.choice(candidates)
        bootstrapped.iloc[idx] = referenced_vectors.iloc[donor]

    print(f"Hybrid Bootstrap: {n_keep}/{n} conservadas ({keep_percentage*100:.0f}%), {n_replace} reemplazadas")

    return reconstruct_ohlc_from_referenced(bootstrapped, initial_price, validate_output=False)


# ---------------------------------------------------------------------------
# 5. Block-Aware Hybrid Bootstrap
# ---------------------------------------------------------------------------

def block_aware_hybrid_bootstrap(
    referenced_vectors: pd.DataFrame,
    initial_price: float,
    block_size: int = 5,
    keep_percentage: float = 0.5,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Divide la serie en bloques de tamaño block_size. Decide por bloques
    completos cuáles conservar (según keep_percentage). Los bloques a
    reemplazar se sustituyen por bloques aleatorios de igual longitud.

    Args:
        referenced_vectors: Vectores OHLC referenciados.
        initial_price: Precio inicial para reconstrucción.
        block_size: Tamaño de cada bloque.
        keep_percentage: Fracción de bloques a conservar.
        seed: Semilla para reproducibilidad.

    Returns:
        DataFrame OHLC sintético reconstruido.
    """
    if seed is not None:
        np.random.seed(seed)

    n = len(referenced_vectors)
    n_blocks = int(np.ceil(n / block_size))
    n_keep_blocks = max(1, int(n_blocks * keep_percentage))

    keep_block_ids = np.sort(np.random.choice(n_blocks, size=n_keep_blocks, replace=False))
    replace_block_ids = [i for i in range(n_blocks) if i not in keep_block_ids]

    bootstrapped = referenced_vectors.copy()

    for bid in replace_block_ids:
        start = bid * block_size
        end = min(start + block_size, n)
        actual_len = end - start

        # Elegir un bloque donante aleatorio
        donor_start = np.random.randint(0, n - actual_len + 1)
        donor_block = referenced_vectors.iloc[donor_start : donor_start + actual_len]

        bootstrapped.iloc[start:end] = donor_block.values

    print(
        f"Block-Aware Hybrid: {n_keep_blocks}/{n_blocks} bloques conservados "
        f"({keep_percentage*100:.0f}%), {len(replace_block_ids)} reemplazados"
    )

    return reconstruct_ohlc_from_referenced(bootstrapped, initial_price, validate_output=False)


# ---------------------------------------------------------------------------
# Generación iterativa de múltiples sintéticas válidas
# ---------------------------------------------------------------------------

def generate_multiple_synthetics(
    referenced_vectors: pd.DataFrame,
    original_data: pd.DataFrame,
    n_synthetics: int = 5,
    method: str = "hybrid",
    keep_percentage: float = 0.5,
    block_size: int = 5,
    ks_target: float = 0.95,
    ret_target: float = 0.80,
    sq_target: float = 0.80,
    max_iterations_per_synthetic: int = 1000,
    price_col: str = "Close",
) -> Dict[str, pd.DataFrame]:
    """
    Genera iterativamente N datasets sintéticos que cumplan umbrales de calidad.

    Para cada sintética, prueba hasta max_iterations_per_synthetic intentos con
    hybrid_bootstrap_ohlc y evalúa con K-S, ACF retornos y ACF retornos².

    Args:
        referenced_vectors: Vectores OHLC referenciados.
        original_data: DataFrame OHLC original (para extraer initial_price y retornos).
        n_synthetics: Número de sintéticas válidas deseadas.
        method: Método de bootstrap ('simple', 'block', 'intra_block', 'hybrid', 'block_aware').
        keep_percentage: Para métodos hybrid.
        block_size: Para métodos de bloque.
        ks_target: Umbral mínimo K-S p-value.
        ret_target: Umbral mínimo correlación ACF retornos.
        sq_target: Umbral mínimo correlación ACF retornos².
        max_iterations_per_synthetic: Intentos máximos por sintética.
        price_col: Columna de precio a usar.

    Returns:
        Diccionario {synth_1: DataFrame, synth_2: DataFrame, ...}.
    """
    initial_price = float(original_data[price_col].iloc[0])
    original_returns = np.log(
        original_data[price_col] / original_data[price_col].shift(1)
    ).dropna().values

    # Seleccionar función de bootstrap
    bootstrap_fn_map = {
        "simple": lambda rv, ip, s: bootstrap_ohlc_simple(rv, ip, seed=s),
        "block": lambda rv, ip, s: block_bootstrap_ohlc(rv, ip, block_size=block_size, seed=s),
        "intra_block": lambda rv, ip, s: intra_block_shuffle_bootstrap(rv, ip, block_size=block_size, seed=s),
        "hybrid": lambda rv, ip, s: hybrid_bootstrap_ohlc(rv, ip, keep_percentage=keep_percentage, seed=s),
        "block_aware": lambda rv, ip, s: block_aware_hybrid_bootstrap(
            rv, ip, block_size=block_size, keep_percentage=keep_percentage, seed=s
        ),
    }

    if method not in bootstrap_fn_map:
        raise ValueError(f"Método desconocido: {method}. Opciones: {list(bootstrap_fn_map.keys())}")

    bootstrap_fn = bootstrap_fn_map[method]

    synthetic_datasets: Dict[str, pd.DataFrame] = {}
    found = 0

    print(f"\n{'='*60}")
    print(f"Generando {n_synthetics} sintéticas válidas (método: {method})")
    print(f"Umbrales — KS: {ks_target}, Ret: {ret_target}, Sq: {sq_target}")
    print(f"{'='*60}\n")

    for synth_id in range(1, n_synthetics + 1):
        print(f"\n--- Buscando sintética {synth_id}/{n_synthetics} ---")
        for iteration in tqdm(range(1, max_iterations_per_synthetic + 1), desc=f"synth_{synth_id}"):
            try:
                synth = bootstrap_fn(referenced_vectors, initial_price, iteration)
                synth_returns = np.log(
                    synth[price_col] / synth[price_col].shift(1)
                ).dropna().values

                if len(synth_returns) < 10:
                    continue

                metrics = evaluate_synthetic_quality(original_returns, synth_returns)

                if (
                    metrics["ks_pvalue"] > ks_target
                    and metrics["ret_correlation"] > ret_target
                    and metrics["sq_correlation"] > sq_target
                ):
                    key = f"synth_{synth_id}"
                    synthetic_datasets[key] = synth
                    found += 1
                    print(
                        f"\n✓ {key} encontrada en iteración {iteration}: "
                        f"KS={metrics['ks_pvalue']:.4f}, "
                        f"Ret={metrics['ret_correlation']:.4f}, "
                        f"Sq={metrics['sq_correlation']:.4f}"
                    )
                    break

                if iteration % 200 == 0:
                    print(
                        f"  iter {iteration}: KS={metrics['ks_pvalue']:.4f}, "
                        f"Ret={metrics['ret_correlation']:.4f}, "
                        f"Sq={metrics['sq_correlation']:.4f}"
                    )

            except Exception as e:
                if iteration % 500 == 0:
                    print(f"  ⚠️ Error en iteración {iteration}: {e}")
                continue
        else:
            print(f"\n✗ No se encontró sintética {synth_id} en {max_iterations_per_synthetic} intentos")

    print(f"\n{'='*60}")
    print(f"Resultado: {found}/{n_synthetics} sintéticas válidas generadas")
    print(f"{'='*60}")

    return synthetic_datasets
