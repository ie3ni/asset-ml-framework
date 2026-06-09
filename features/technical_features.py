import pandas as pd
import numpy as np

def build_features(
        df,
        cfg):

    df['return_1d'] = df['close'].pct_change()

    df['momentum_fast'] = (
        df['close']
        .pct_change(cfg.MOMENTUM_FAST)
    )

    df['momentum_slow'] = (
        df['close']
        .pct_change(cfg.MOMENTUM_SLOW)
    )

    df['vol_fast'] = (
        df['return_1d']
        .rolling(cfg.VOL_FAST)
        .std()
    )

    df['vol_slow'] = (
        df['return_1d']
        .rolling(cfg.VOL_SLOW)
        .std()
    )

    df['sma_fast'] = (
        df['close']
        .rolling(cfg.SMA_FAST)
        .mean()
    )

    df['sma_slow'] = (
        df['close']
        .rolling(cfg.SMA_SLOW)
        .mean()
    )

    df['ma_spread'] = (
        df['sma_fast']
        -
        df['sma_slow']
    )

    df['volume_change'] = (
        df['volume']
        .pct_change()
    )

    delta = df['close'].diff()

    gain = np.where(
        delta > 0,
        delta,
        0
    )

    loss = np.where(
        delta < 0,
        -delta,
        0
    )

    gain = pd.Series(
        gain,
        index=df.index
    )

    loss = pd.Series(
        loss,
        index=df.index
    )

    avg_gain = gain.rolling(
        cfg.RSI_WINDOW
    ).mean()

    avg_loss = loss.rolling(
        cfg.RSI_WINDOW
    ).mean()

    rs = avg_gain / avg_loss

    df['rsi'] = (
        100
        -
        (100 / (1 + rs))
    )

    df['target'] = (
        df['close']
        .shift(-1)
        >
        df['close']
    ).astype(int)

    df['today_direction'] = (
        df['close']
        >
        df['close'].shift(1)
    ).astype(int)

    df.dropna(inplace=True)

    return df