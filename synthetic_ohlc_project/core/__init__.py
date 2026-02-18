"""
SyntheticData QXPRO vQDemy — Core modules
Generador de data sintética OHLC con bootstrap y evaluación estadística.
"""

from .ohlc_reference import (
    parse_ohlc_to_referenced,
    reconstruct_ohlc_from_referenced,
    get_summary_statistics,
    plot_referenced_vectors_histograms,
)
from .bootstrap_methods import (
    bootstrap_ohlc_simple,
    block_bootstrap_ohlc,
    intra_block_shuffle_bootstrap,
    hybrid_bootstrap_ohlc,
    block_aware_hybrid_bootstrap,
    generate_multiple_synthetics,
)
from .evaluation import evaluate_synthetic_quality, compare_synthetic_data
from .data_loader import load_csv_data, split_dataset
from .monkey_filters import (
    plot_histogram_simulacion_oos,
    rendimientos_primer_ultimo_dia,
    filtrar_años_positivos,
    evaluar_reglas_años_positivos,
    meta_regla_simple,
)
