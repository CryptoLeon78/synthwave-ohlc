"""
core/data_loader.py
Carga de CSV OHLC, creación de Target y partición IS/OOS/Forward.
"""

import pandas as pd
from typing import Tuple, Optional


def load_csv_data(
    filepath: str,
    date_col: str = "Date",
    parse_dates: bool = True,
) -> pd.DataFrame:
    """
    Carga un CSV con datos OHLC y prepara el DataFrame.

    Args:
        filepath: Ruta al archivo CSV.
        date_col: Nombre de la columna de fecha.
        parse_dates: Si convertir a datetime.

    Returns:
        DataFrame con Date como índice y columnas OHLC.

    Raises:
        ValueError: Si faltan columnas esenciales.
    """
    df = pd.read_csv(filepath)

    if date_col not in df.columns:
        raise ValueError(f"Columna '{date_col}' no encontrada. Columnas disponibles: {list(df.columns)}")

    if parse_dates:
        df[date_col] = pd.to_datetime(df[date_col])

    df = df.set_index(date_col).sort_index()

    required = ["Open", "High", "Low", "Close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas OHLC requeridas: {missing}")

    # Crear Target = (Open.shift(-2) - Open.shift(-1)) / Open.shift(-1)
    df["Target"] = (df["Open"].shift(-2) - df["Open"].shift(-1)) / df["Open"].shift(-1)

    print(f"✓ Datos cargados: {len(df)} filas, rango {df.index.min()} → {df.index.max()}")

    return df


def split_dataset(
    df: pd.DataFrame,
    is_end: str,
    oos_end: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Divide el dataset en In-Sample, Out-of-Sample y Forward.

    Args:
        df: DataFrame completo con índice datetime.
        is_end: Fecha fin del periodo In-Sample (inclusive).
        oos_end: Fecha fin del periodo OOS (inclusive); el resto es Forward.

    Returns:
        Tupla (df_is, df_oos, df_forward).
    """
    df_is = df.loc[:is_end].copy()
    df_oos = df.loc[is_end:oos_end].iloc[1:].copy()  # excluir fecha de corte
    df_forward = df.loc[oos_end:].iloc[1:].copy()

    print(f"✓ IS: {len(df_is)} | OOS: {len(df_oos)} | Forward: {len(df_forward)}")

    return df_is, df_oos, df_forward
