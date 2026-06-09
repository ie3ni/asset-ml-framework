import yfinance as yf

def download_data(
        ticker,
        period,
        interval):

    print(
        f"Downloading {ticker}..."
    )

    df = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=True
    )

    return df