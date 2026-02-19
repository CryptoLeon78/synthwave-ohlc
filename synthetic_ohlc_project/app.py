"""
SyntheticData QXPRO vQDemy
=========================
Generador de data sintética OHLC con bootstrap y evaluación estadística.
Replica la lógica de "Sesión 4: Data Sintética, Ensamblado y Filtros" (PyRE by Quantdemy).

Ejecutar:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
import zipfile
import matplotlib
matplotlib.use("Agg")

from core.data_loader import load_csv_data, split_dataset
from core.ohlc_reference import (
    parse_ohlc_to_referenced,
    reconstruct_ohlc_from_referenced,
    get_summary_statistics,
    plot_referenced_vectors_histograms,
)
from core.bootstrap_methods import (
    bootstrap_ohlc_simple,
    block_bootstrap_ohlc,
    intra_block_shuffle_bootstrap,
    hybrid_bootstrap_ohlc,
    block_aware_hybrid_bootstrap,
    generate_multiple_synthetics,
)
from core.evaluation import evaluate_synthetic_quality, compare_synthetic_data
from core.monkey_filters import (
    plot_histogram_simulacion_oos,
    rendimientos_primer_ultimo_dia,
    filtrar_años_positivos,
    evaluar_reglas_años_positivos,
    meta_regla_simple,
)
from core.ensemble import (
    ensemble_mean,
    ensemble_median,
    ensemble_returns,
    ensemble_target_signals,
    plot_ensemble_comparison,
    compute_ensemble_bands,
    plot_ensemble_bands,
)
from core.adverse_test import adverse_test_returns, adverse_test_monkey

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="SyntheticData QXPRO vQDemy",
    page_icon="📈",
    layout="wide",
)

st.title("📈 SyntheticData QXPRO vQDemy")
st.caption("Generador de data sintética OHLC — Bootstrap & Evaluación Estadística")

# ──────────────────────────────────────────────
# Sidebar: Upload & config
# ──────────────────────────────────────────────
with st.sidebar:
    st.header("1 · Datos de entrada")
    uploaded_file = st.file_uploader("Subir CSV con OHLC", type=["csv"])

    st.markdown("---")
    st.header("2 · Partición temporal")
    is_end = st.text_input("Fin In-Sample (YYYY-MM-DD)", value="2018-12-31")
    oos_end = st.text_input("Fin Out-of-Sample (YYYY-MM-DD)", value="2021-12-31")
    date_col = st.text_input("Columna de fecha", value="Date")
    price_col = st.selectbox("Columna de precio para métricas", ["Close", "Open"], index=0)
    handle_missing = st.selectbox("Missing values", ["drop", "interpolate", "fill_forward"])

# ──────────────────────────────────────────────
# Load data
# ──────────────────────────────────────────────
if uploaded_file is None:
    st.info("⬆️ Sube un archivo CSV con columnas: Date, Open, High, Low, Close")
    st.stop()

try:
    raw_df = pd.read_csv(uploaded_file)
    if date_col in raw_df.columns:
        raw_df[date_col] = pd.to_datetime(raw_df[date_col])
        raw_df = raw_df.set_index(date_col).sort_index()
    raw_df["Target"] = (raw_df["Open"].shift(-2) - raw_df["Open"].shift(-1)) / raw_df["Open"].shift(-1)
except Exception as e:
    st.error(f"Error leyendo CSV: {e}")
    st.stop()

st.success(f"✓ {len(raw_df)} filas cargadas — {raw_df.index.min()} → {raw_df.index.max()}")

try:
    df_is, df_oos, df_forward = split_dataset(raw_df, is_end, oos_end)
except Exception as e:
    st.warning(f"No se pudo partir: {e}. Usando todo como IS.")
    df_is = raw_df
    df_oos = raw_df.iloc[:0]
    df_forward = raw_df.iloc[:0]

col_a, col_b, col_c = st.columns(3)
col_a.metric("In-Sample", f"{len(df_is)} filas")
col_b.metric("Out-of-Sample", f"{len(df_oos)} filas")
col_c.metric("Forward", f"{len(df_forward)} filas")

# ──────────────────────────────────────────────
# Tabs
# ──────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Vectores Referenciados",
    "🔄 Bootstrap Individual",
    "🚀 Generación Masiva",
    "🧩 Ensemble",
    "⚔️ Test Adverso",
    "🐒 Monkey Test",
    "📅 Análisis Anual & Filtros",
])

# ──────────────────────────────────────────────
# Tab 1: Referenced vectors
# ──────────────────────────────────────────────
with tab1:
    st.subheader("Conversión a OHLC Referenciado")
    try:
        ref_vectors = parse_ohlc_to_referenced(df_is, handle_missing=handle_missing)
        st.session_state["ref_vectors"] = ref_vectors
        st.session_state["df_is"] = df_is

        stats_df = get_summary_statistics(ref_vectors)
        st.dataframe(stats_df.style.format("{:.6f}"), use_container_width=True)

        fig = plot_referenced_vectors_histograms(ref_vectors)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error: {e}")

# ──────────────────────────────────────────────
# Tab 2: Single bootstrap
# ──────────────────────────────────────────────
with tab2:
    st.subheader("Bootstrap individual")

    if "ref_vectors" not in st.session_state:
        st.warning("Primero genera los vectores referenciados en la pestaña anterior.")
        st.stop()

    ref_vectors = st.session_state["ref_vectors"]
    df_is_local = st.session_state["df_is"]

    c1, c2, c3, c4 = st.columns(4)
    method = c1.selectbox("Método", ["simple", "block", "intra_block", "hybrid", "block_aware"])
    seed = c2.number_input("Seed", value=42, step=1)
    block_size = c3.number_input("Block size", value=5, min_value=2, step=1)
    keep_pct = c4.slider("Keep %", 0.1, 0.9, 0.5, 0.05)
    initial_price = st.number_input("Precio inicial", value=float(df_is_local[price_col].iloc[0]), format="%.5f")

    if st.button("Generar sintética", key="gen_single"):
        with st.spinner("Generando..."):
            try:
                fn_map = {
                    "simple": lambda: bootstrap_ohlc_simple(ref_vectors, initial_price, seed=seed),
                    "block": lambda: block_bootstrap_ohlc(ref_vectors, initial_price, block_size=block_size, seed=seed),
                    "intra_block": lambda: intra_block_shuffle_bootstrap(ref_vectors, initial_price, block_size=block_size, seed=seed),
                    "hybrid": lambda: hybrid_bootstrap_ohlc(ref_vectors, initial_price, keep_percentage=keep_pct, seed=seed),
                    "block_aware": lambda: block_aware_hybrid_bootstrap(ref_vectors, initial_price, block_size=block_size, keep_percentage=keep_pct, seed=seed),
                }
                synth = fn_map[method]()
                st.session_state["last_synth"] = synth

                fig, metrics = compare_synthetic_data(df_is_local, synth, price_col=price_col)
                st.pyplot(fig)

                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("K-S p-value", f"{metrics['ks_pvalue']:.4f}")
                mc2.metric("ACF Retornos", f"{metrics['ret_correlation']:.4f}")
                mc3.metric("ACF Retornos²", f"{metrics['sq_correlation']:.4f}")

                csv_buf = synth.to_csv()
                st.download_button("📥 Descargar CSV", csv_buf, file_name=f"synthetic_{method}_{seed}.csv", mime="text/csv")

            except Exception as e:
                st.error(f"Error: {e}")

# ──────────────────────────────────────────────
# Tab 3: Multiple synthetics
# ──────────────────────────────────────────────
with tab3:
    st.subheader("Generación iterativa de múltiples sintéticas")

    if "ref_vectors" not in st.session_state:
        st.warning("Primero genera los vectores referenciados.")
        st.stop()

    ref_vectors = st.session_state["ref_vectors"]
    df_is_local = st.session_state["df_is"]

    c1, c2 = st.columns(2)
    with c1:
        n_synth = st.number_input("Nº sintéticas", value=5, min_value=1, max_value=50, step=1)
        m_method = st.selectbox("Método bootstrap", ["hybrid", "simple", "block", "intra_block", "block_aware"], key="mass_method")
        m_keep = st.slider("Keep %", 0.1, 0.9, 0.5, 0.05, key="mass_keep")
        m_block = st.number_input("Block size", value=5, min_value=2, step=1, key="mass_block")
    with c2:
        ks_t = st.number_input("Umbral KS p-value", value=0.95, min_value=0.0, max_value=1.0, step=0.05)
        ret_t = st.number_input("Umbral ACF Ret", value=0.80, min_value=0.0, max_value=1.0, step=0.05)
        sq_t = st.number_input("Umbral ACF Ret²", value=0.80, min_value=0.0, max_value=1.0, step=0.05)
        max_iter = st.number_input("Max iteraciones/sintética", value=1000, min_value=100, step=100)

    if st.button("🚀 Generar múltiples", key="gen_mass"):
        with st.spinner(f"Buscando {n_synth} sintéticas válidas... (puede tardar)"):
            try:
                results = generate_multiple_synthetics(
                    ref_vectors, df_is_local,
                    n_synthetics=n_synth,
                    method=m_method,
                    keep_percentage=m_keep,
                    block_size=m_block,
                    ks_target=ks_t,
                    ret_target=ret_t,
                    sq_target=sq_t,
                    max_iterations_per_synthetic=max_iter,
                    price_col=price_col,
                )
                st.session_state["mass_results"] = results
                st.success(f"✓ {len(results)}/{n_synth} sintéticas válidas generadas")
            except Exception as e:
                st.error(f"Error: {e}")

    if "mass_results" in st.session_state and st.session_state["mass_results"]:
        results = st.session_state["mass_results"]
        selected = st.selectbox("Seleccionar sintética", list(results.keys()))

        if selected:
            synth = results[selected]
            fig, metrics = compare_synthetic_data(df_is_local, synth, price_col=price_col)
            st.pyplot(fig)

            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("K-S p-value", f"{metrics['ks_pvalue']:.4f}")
            mc2.metric("ACF Retornos", f"{metrics['ret_correlation']:.4f}")
            mc3.metric("ACF Retornos²", f"{metrics['sq_correlation']:.4f}")

            csv_buf = synth.to_csv()
            st.download_button(f"📥 Descargar {selected}", csv_buf, file_name=f"{selected}.csv", mime="text/csv")

# ──────────────────────────────────────────────
# Tab 4: Ensemble
# ──────────────────────────────────────────────
with tab4:
    st.subheader("🧩 Ensamblado (Ensemble) de Sintéticas")

    if "mass_results" not in st.session_state or not st.session_state["mass_results"]:
        st.warning("Primero genera múltiples sintéticas en la pestaña 'Generación Masiva'.")
    else:
        results = st.session_state["mass_results"]
        df_is_local = st.session_state["df_is"]

        st.info(f"Sintéticas disponibles para ensamblar: {len(results)} → {list(results.keys())}")

        c1, c2 = st.columns(2)
        ens_method = c1.selectbox("Método de ensemble", ["mean", "median", "returns"])
        lower_pct = c2.slider("Percentil inferior (bandas)", 1.0, 25.0, 5.0, 1.0)
        upper_pct = c2.slider("Percentil superior (bandas)", 75.0, 99.0, 95.0, 1.0)

        if st.button("🧩 Construir Ensemble", key="build_ensemble"):
            with st.spinner("Ensamblando..."):
                try:
                    if ens_method == "mean":
                        ensemble_df = ensemble_mean(results)
                    elif ens_method == "median":
                        ensemble_df = ensemble_median(results)
                    else:
                        init_p = float(df_is_local[price_col].iloc[0])
                        ensemble_df = ensemble_returns(results, init_p, price_col=price_col)

                    st.session_state["ensemble_df"] = ensemble_df

                    fig, metrics = plot_ensemble_comparison(
                        df_is_local, results, ensemble_df, price_col=price_col,
                    )
                    st.pyplot(fig)

                    mc1, mc2, mc3 = st.columns(3)
                    mc1.metric("K-S p-value", f"{metrics['ks_pvalue']:.4f}")
                    mc2.metric("ACF Retornos", f"{metrics['ret_correlation']:.4f}")
                    mc3.metric("ACF Retornos²", f"{metrics['sq_correlation']:.4f}")

                    # Bandas de confianza
                    st.markdown("---")
                    st.subheader("Bandas de Confianza")
                    bands = compute_ensemble_bands(results, price_col=price_col,
                                                   lower_pct=lower_pct, upper_pct=upper_pct)
                    fig_bands = plot_ensemble_bands(df_is_local, bands, price_col=price_col)
                    st.pyplot(fig_bands)

                    # Descargar ensemble
                    csv_ens = ensemble_df.to_csv()
                    st.download_button(
                        "📥 Descargar Ensemble CSV", csv_ens,
                        file_name=f"ensemble_{ens_method}.csv", mime="text/csv",
                    )

                except Exception as e:
                    st.error(f"Error: {e}")

        # ── Votación de señales Target ──
        st.markdown("---")
        st.subheader("🗳️ Votación de Señales Target")
        st.caption("Combina las predicciones Target de múltiples sintéticas por votación.")

        vote_threshold = st.slider("Umbral de consenso (proporción mínima)", 0.1, 0.9, 0.5, 0.05, key="vote_thresh")

        if st.button("🗳️ Calcular Votación", key="calc_vote"):
            with st.spinner("Calculando votación..."):
                try:
                    # Añadir Target a sintéticas si no existe
                    results_with_target = {}
                    for k, sdf in results.items():
                        sdf_copy = sdf.copy()
                        if "Target" not in sdf_copy.columns:
                            sdf_copy["Target"] = (sdf_copy["Open"].shift(-2) - sdf_copy["Open"].shift(-1)) / sdf_copy["Open"].shift(-1)
                        results_with_target[k] = sdf_copy

                    signals_df = ensemble_target_signals(results_with_target, threshold=vote_threshold)
                    st.session_state["ensemble_signals"] = signals_df

                    c1, c2, c3 = st.columns(3)
                    n_pos = int(signals_df["ensemble_signal"].sum())
                    c1.metric("Señales positivas", f"{n_pos}/{len(signals_df)}")
                    c2.metric("Ratio medio votos", f"{signals_df['vote_ratio'].mean():.3f}")
                    c3.metric("Target medio ensemble", f"{signals_df['avg_target'].mean():.6f}")

                    st.dataframe(signals_df.head(20), use_container_width=True)

                    # Rendimiento con señal
                    if "Target" in df_is_local.columns:
                        aligned = signals_df["ensemble_signal"].reindex(df_is_local.index).fillna(0)
                        ret_signal = df_is_local["Target"].fillna(0) * aligned
                        cum_ret = float((1 + ret_signal).prod() - 1)
                        st.metric("Rendimiento acumulado con señal de votación", f"{cum_ret:.4f}")

                except Exception as e:
                    st.error(f"Error: {e}")

        # ── ZIP download ──
        st.markdown("---")
        st.subheader("📦 Descargar Todo en ZIP")

        if st.button("📦 Generar ZIP", key="gen_zip"):
            with st.spinner("Creando archivo ZIP..."):
                try:
                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                        # Sintéticas
                        for name, sdf in results.items():
                            zf.writestr(f"{name}.csv", sdf.to_csv())

                        # Ensemble
                        if "ensemble_df" in st.session_state:
                            zf.writestr(f"ensemble_{ens_method}.csv", st.session_state["ensemble_df"].to_csv())

                        # Señales de votación
                        if "ensemble_signals" in st.session_state:
                            zf.writestr("ensemble_signals.csv", st.session_state["ensemble_signals"].to_csv())

                    zip_buffer.seek(0)
                    st.download_button(
                        "📥 Descargar ZIP completo",
                        zip_buffer.getvalue(),
                        file_name="synthetic_ohlc_bundle.zip",
                        mime="application/zip",
                    )
                    st.success(f"✓ ZIP con {len(results)} sintéticas listo para descarga")

                except Exception as e:
                    st.error(f"Error: {e}")

# ──────────────────────────────────────────────
# Tab 5: Adverse Test
# ──────────────────────────────────────────────
with tab5:
    st.subheader("⚔️ Test Adverso de Robustez")

    if "mass_results" not in st.session_state or not st.session_state["mass_results"]:
        st.warning("Primero genera múltiples sintéticas en la pestaña 'Generación Masiva'.")
    else:
        results = st.session_state["mass_results"]
        df_is_local = st.session_state["df_is"]

        st.markdown("### Test Adverso por Señal")
        st.caption("Evalúa si una señal/estrategia es robusta comparando rendimientos en original vs sintéticas.")

        adv_rules = st.text_area(
            "Reglas de señal (formato pandas query, una por línea)",
            value="close_relative > 0",
            height=80,
            key="adv_rules",
        )

        if st.button("⚔️ Ejecutar Test Adverso", key="run_adverse"):
            with st.spinner("Ejecutando test adverso..."):
                try:
                    # Generar señal desde reglas
                    if "ref_vectors" in st.session_state:
                        eval_df = st.session_state["ref_vectors"].copy()
                        eval_df["Target"] = df_is_local["Target"].reindex(eval_df.index)
                        eval_df = eval_df.dropna(subset=["Target"])

                        rules = [r.strip() for r in adv_rules.strip().split("\n") if r.strip()]
                        signal = pd.Series(False, index=eval_df.index)
                        for rule in rules:
                            try:
                                signal = signal | eval_df.eval(rule)
                            except Exception:
                                pass
                        signal = signal.astype(int)

                        # Necesitamos original con Target
                        orig_with_target = df_is_local.copy()

                        # Añadir Target a sintéticas
                        results_t = {}
                        for k, sdf in results.items():
                            sc = sdf.copy()
                            if "Target" not in sc.columns:
                                sc["Target"] = (sc["Open"].shift(-2) - sc["Open"].shift(-1)) / sc["Open"].shift(-1)
                            results_t[k] = sc

                        fig, stats = adverse_test_returns(
                            orig_with_target, results_t, signal,
                        )
                        st.pyplot(fig)

                        c1, c2, c3 = st.columns(3)
                        c1.metric("Rendimiento Original", f"{stats['original_return']:.4f}")
                        c2.metric("Percentil", f"{stats['percentile_rank']:.1f}%")
                        c3.metric("Veredicto", "✅ Robusta" if stats["is_robust"] else "❌ No robusta")
                    else:
                        st.warning("Genera vectores referenciados primero.")
                except Exception as e:
                    st.error(f"Error: {e}")

        st.markdown("---")
        st.markdown("### Test Adverso Monkey")
        st.caption("Genera señales aleatorias y compara comportamiento en original vs sintéticas.")

        c1, c2 = st.columns(2)
        n_rand = c1.number_input("Nº señales aleatorias", value=500, min_value=50, step=50, key="n_rand_adv")
        sig_dens = c2.slider("Densidad de señal", 0.1, 0.9, 0.5, 0.05, key="sig_dens")

        if st.button("🐒 Ejecutar Monkey Adverso", key="run_monkey_adv"):
            with st.spinner("Simulando..."):
                try:
                    orig_t = df_is_local.copy()
                    results_t = {}
                    for k, sdf in results.items():
                        sc = sdf.copy()
                        if "Target" not in sc.columns:
                            sc["Target"] = (sc["Open"].shift(-2) - sc["Open"].shift(-1)) / sc["Open"].shift(-1)
                        results_t[k] = sc

                    fig, stats = adverse_test_monkey(
                        orig_t, results_t,
                        n_random_signals=n_rand,
                        signal_density=sig_dens,
                    )
                    st.pyplot(fig)

                    c1, c2 = st.columns(2)
                    c1.metric("Correlación Orig↔Sint", f"{stats['correlation_orig_synth']:.3f}")
                    c2.metric("Comportamiento similar", "✅ Sí" if stats["similar_behavior"] else "❌ No")
                except Exception as e:
                    st.error(f"Error: {e}")

# ──────────────────────────────────────────────
# Tab 6: Monkey Test
# ──────────────────────────────────────────────
with tab6:
    st.subheader("Monkey Test — Simulación OOS")

    if len(df_oos) < 5:
        st.warning("El periodo Out-of-Sample tiene pocas filas. Ajusta las fechas de corte.")
    else:
        c1, c2, c3 = st.columns(3)
        n_sim = c1.number_input("Nº simulaciones", value=1000, min_value=100, step=100)
        s_frac = c2.slider("Fracción muestra", 0.1, 1.0, 0.5, 0.05)
        q_val = c3.number_input("Quantil (%)", value=80, min_value=1, max_value=99)

        if st.button("Ejecutar Monkey Test"):
            with st.spinner("Simulando..."):
                try:
                    fig, stats = plot_histogram_simulacion_oos(
                        df_oos, n_simulations=n_sim, sample_fraction=s_frac, quantile=q_val,
                    )
                    st.pyplot(fig)
                    st.json(stats)
                except Exception as e:
                    st.error(f"Error: {e}")

# ──────────────────────────────────────────────
# Tab 7: Annual analysis & filters
# ──────────────────────────────────────────────
with tab7:
    st.subheader("Rendimientos anuales y filtros")

    try:
        annual, annual_stats = rendimientos_primer_ultimo_dia(raw_df, price_col=price_col)
        st.dataframe(annual.style.format("{:.4f}"), use_container_width=True)

        ac1, ac2, ac3 = st.columns(3)
        ac1.metric("Media anual", f"{annual_stats['media']:.2f}%")
        ac2.metric("% Años positivos", f"{annual_stats['pct_positivos']:.1f}%")
        ac3.metric("Mejor / Peor", f"{annual_stats['mejor_año']} / {annual_stats['peor_año']}")

        st.markdown("---")
        st.subheader("Filtro: solo años positivos")

        df_pos, pos_years = filtrar_años_positivos(raw_df, annual, price_col=price_col)
        st.write(f"Años positivos: {pos_years}")
        st.write(f"Filas filtradas: {len(df_pos)}")

        st.markdown("---")
        st.subheader("Evaluación de reglas")
        rules_text = st.text_area(
            "Reglas (una por línea, formato pandas query)",
            value="close_relative > 0\nopen_relative < 0",
            height=100,
        )
        if st.button("Evaluar reglas"):
            rules = [r.strip() for r in rules_text.strip().split("\n") if r.strip()]
            if "ref_vectors" in st.session_state:
                eval_df = st.session_state["ref_vectors"].copy()
                eval_df["Target"] = df_is_local["Target"].reindex(eval_df.index)
                eval_df = eval_df.dropna(subset=["Target"])
                result = evaluar_reglas_años_positivos(eval_df, rules)
                st.dataframe(result, use_container_width=True)
            else:
                st.warning("Genera los vectores referenciados primero.")

    except Exception as e:
        st.error(f"Error: {e}")
