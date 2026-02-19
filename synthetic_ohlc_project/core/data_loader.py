"""
core/data_loader.py
Carga de CSV OHLC, creación de Target y partición IS/OOS/Forward.
Compatible con exportaciones MT5 (separador tab, columnas <DATE>/<TIME>).
"""

import pandas as pd
from typing import Tuple, Optional


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza nombres de columnas para compatibilidad con MT5 y otros brokers.
    - Elimina espacios y caracteres < >
    - Capitaliza (open→Open, close→Close...)
    - Renombra aliases comunes (time→Date, tickvol→Volume...)
    """
    df.columns = df.columns.str.strip()
    df.columns = df.columns.str.replace(r'[<>]', '', regex=True)
    df.columns = df.columns.str.strip()
    df.columns = df.columns.str.capitalize()

    rename_map = {
        "Tickvol": "Volume",
        "Vol":     "Volume",
    }
    df.rename(columns=rename_map, inplace=True)
    return df


def _merge_date_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Si existen columnas 'Date' y 'Time' separadas (formato MT5),
    las combina en una sola columna 'Date' datetime y elimina 'Time'.
    """
    if "Date" in df.columns and "Time" in df.columns:
        df["Date"] = pd.to_datetime(
            df["Date"].astype(str) + " " + df["Time"].astype(str)
        )
        df.drop(columns=["Time"], inplace=True)
        print("✓ Columnas Date + Time combinadas en un solo datetime")
    return df


def load_csv_data(
    filepath: str,
    date_col: str = "Date",
    parse_dates: bool = True,
) -> pd.DataFrame:
    """
    Carga un CSV con datos OHLC y prepara el DataFrame.
    Compatible con exportaciones de MT5 (tab-separated, <DATE>+<TIME>).

    Args:
        filepath: Ruta al archivo CSV.
        date_col: Nombre de la columna de fecha (tras normalización).
        parse_dates: Si convertir a datetime.

    Returns:
        DataFrame con Date como índice y columnas OHLC.

    Raises:
        ValueError: Si faltan columnas esenciales.
    """
    # Detectar separador automáticamente (tab o coma)
    with open(filepath, "r") as f:
        first_line = f.readline()
    sep = "\t" if "\t" in first_line else ","

    df = pd.read_csv(filepath, sep=sep)

    # Normalizar columnas
    df = _normalize_columns(df)

    # Combinar Date + Time si vienen separadas (MT5)
    df = _merge_date_time(df)

    if date_col not in df.columns:
        raise ValueError(
            f"Columna '{date_col}' no encontrada tras normalización. "
            f"Columnas disponibles: {list(df.columns)}"
        )

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
    """
    df_is = df.loc[:is_end].copy()
    df_oos = df.loc[is_end:oos_end].iloc[1:].copy()
    df_forward = df.loc[oos_end:].iloc[1:].copy()

    print(f"✓ IS: {len(df_is)} | OOS: {len(df_oos)} | Forward: {len(df_forward)}")

    return df_is, df_oos, df_forward
