from __future__ import annotations

from pathlib import Path

import pandas as pd
from binance.client import Client


SYMBOL = "BTCUSDT"
INTERVAL = "1h"
START_STR = "1 Jan, 2021"
END_STR = "1 Jan, 2022"
OUTPUT_CSV = Path("data/raw/binance_BTCUSDT_1h_2021.csv")


def _klines_to_df(klines) -> pd.DataFrame:
    columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]
    if not klines:
        return pd.DataFrame(columns=columns)

    if len(klines[0]) >= 6:
        rows = [[k[1], k[2], k[3], k[4], k[5]] for k in klines]
    elif len(klines[0]) == 5:
        rows = klines
    else:
        raise ValueError(
            f"Unexpected kline length {len(klines[0])}; expected 5 or >=6."
        )

    df = pd.DataFrame(rows, columns=columns)

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    return df


def main() -> None:
    client = Client()

    klines = client.get_historical_klines(
        SYMBOL,
        INTERVAL,
        START_STR,
        END_STR,
    )

    df = _klines_to_df(klines)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {len(df)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
