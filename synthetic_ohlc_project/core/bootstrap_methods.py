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
    el método seleccionado y evalúa con K-S, ACF retornos y ACF retornos².

    Args:
        referenced_vectors: Vectores OHLC referenciados.
        original_data: DataFrame OHLC original.
        n_synthetics: Número de sintéticas válidas deseadas.
        method: Método de bootstrap.
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

    bootstrap_fn_map = {
        "simple":      lambda rv, ip, s: bootstrap_ohlc_simple(rv, ip, seed=s),
        "block":       lambda rv, ip, s: block_bootstrap_ohlc(rv, ip, block_size=block_size, seed=s),
        "intra_block": lambda rv, ip, s: intra_block_shuffle_bootstrap(rv, ip, block_size=block_size, seed=s),
        "hybrid":      lambda rv, ip, s: hybrid_bootstrap_ohlc(rv, ip, keep_percentage=keep_percentage, seed=s),
        "block_aware": lambda rv, ip, s: block_aware_hybrid_bootstrap(
            rv, ip, block_size=block_size, keep_percentage=keep_percentage, seed=s
        ),
    }

    if method not in bootstrap_fn_map:
        raise ValueError(f"Método desconocido: {method}. Opciones: {list(bootstrap_fn_map.keys())}")

    bootstrap_fn = bootstrap_fn_map[method]

    synthetic_datasets: Dict[str, pd.DataFrame] = {}
    found = 0

    # ── FIX: seed global incremental para garantizar unicidad entre sintéticas ──
    global_seed = 0

    print(f"\n{'='*60}")
    print(f"Generando {n_synthetics} sintéticas válidas (método: {method})")
    print(f"Umbrales — KS: {ks_target}, Ret: {ret_target}, Sq: {sq_target}")
    print(f"{'='*60}\n")

    for synth_id in range(1, n_synthetics + 1):
        print(f"\n--- Buscando sintética {synth_id}/{n_synthetics} ---")

        for iteration in tqdm(range(1, max_iterations_per_synthetic + 1), desc=f"synth_{synth_id}"):
            # ── FIX: seed única y nunca repetida entre todas las iteraciones ──
            current_seed = global_seed
            global_seed += 1

            try:
                synth = bootstrap_fn(referenced_vectors, initial_price, current_seed)
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
                        f"\n✓ {key} encontrada en iteración {iteration} (seed={current_seed}): "
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
