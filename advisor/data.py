"""
データ収集モジュール
- yfinance: 株価・テクニカル指標
- J-Quants API: 決算・ファンダメンタル
- Yahoo!ファイナンスRSS: 最新ニュース
"""

import os
import math
import logging
import requests
import feedparser
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

# ── J-Quants ────────────────────────────────────────────────────────────────

JQUANTS_BASE = "https://api.jquants.com/v1"

def _jquants_id_token() -> Optional[str]:
    """J-Quants リフレッシュトークンからIDトークンを取得する"""
    refresh_token = os.getenv("JQUANTS_REFRESH_TOKEN", "")
    if not refresh_token or refresh_token == "your-jquants-token-here":
        return None
    try:
        res = requests.post(
            f"{JQUANTS_BASE}/token/auth_refresh",
            params={"refreshtoken": refresh_token},
            timeout=10,
        )
        res.raise_for_status()
        return res.json().get("idToken")
    except Exception as e:
        logger.warning(f"J-Quants IDトークン取得失敗: {e}")
        return None


def get_fundamental_data(ticker: str) -> dict:
    """
    J-Quants APIから決算・ファンダメンタルデータを取得する。
    ticker例: "7203" (トヨタ、.T サフィックスなし)

    取得項目:
    - 直近決算（売上・営業利益・純利益）
    - 業績予想との乖離率
    - PER・PBR・配当利回り
    """
    # ticker から ".T" を除去
    code = ticker.replace(".T", "")

    id_token = _jquants_id_token()
    if not id_token:
        return {"error": "J-Quants APIトークン未設定", "ticker": ticker}

    headers = {"Authorization": f"Bearer {id_token}"}

    try:
        # 財務データ取得
        res = requests.get(
            f"{JQUANTS_BASE}/fins/statements",
            params={"code": code},
            headers=headers,
            timeout=15,
        )
        res.raise_for_status()
        statements = res.json().get("statements", [])

        if not statements:
            return {"error": "決算データなし", "ticker": ticker}

        latest = statements[-1]

        # 予想との乖離計算（売上）
        net_sales = float(latest.get("NetSales") or 0)
        forecast_sales = float(latest.get("ForecastNetSales") or 0)
        sales_deviation = (
            round((net_sales - forecast_sales) / forecast_sales * 100, 2)
            if forecast_sales
            else None
        )

        # 予想との乖離計算（営業利益）
        op_profit = float(latest.get("OperatingProfit") or 0)
        forecast_op = float(latest.get("ForecastOperatingProfit") or 0)
        op_deviation = (
            round((op_profit - forecast_op) / forecast_op * 100, 2)
            if forecast_op
            else None
        )

        return {
            "ticker": ticker,
            "fiscal_year": latest.get("FiscalYear"),
            "fiscal_quarter": latest.get("TypeOfCurrentPeriod"),
            "net_sales": net_sales,
            "operating_profit": op_profit,
            "net_income": float(latest.get("Profit") or 0),
            "forecast_net_sales": forecast_sales,
            "forecast_operating_profit": forecast_op,
            "sales_deviation_pct": sales_deviation,
            "op_profit_deviation_pct": op_deviation,
            "per": latest.get("PriceEarningsRatio"),
            "pbr": latest.get("PriceBookValueRatio"),
            "dividend_yield": latest.get("DividendYield"),
        }

    except Exception as e:
        logger.error(f"J-Quants fundamental取得エラー ({ticker}): {e}")
        return {"error": str(e), "ticker": ticker}


# ── yfinance ────────────────────────────────────────────────────────────────

def _calc_rsi(series: pd.Series, period: int = 14) -> float:
    """RSIを計算する"""
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    val = rsi.iloc[-1]
    return round(float(val), 2) if not math.isnan(val) else None


def _calc_macd(series: pd.Series) -> dict:
    """MACD・シグナル・ヒストグラムを計算する"""
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal = macd_line.ewm(span=9, adjust=False).mean()
    hist = macd_line - signal

    def _v(s):
        v = s.iloc[-1]
        return round(float(v), 4) if not math.isnan(v) else None

    return {"macd": _v(macd_line), "signal": _v(signal), "histogram": _v(hist)}


def get_stock_data(ticker: str) -> dict:
    """
    yfinanceから株価・テクニカル指標を取得する。
    ticker例: "7203.T"

    取得項目:
    - 直近20日の終値・出来高
    - 52週高値・安値
    - 移動平均（5日・25日・75日）
    - RSI（14日）
    - MACD
    - 出来高移動平均比
    """
    try:
        stock = yf.Ticker(ticker)
        # 75日MA計算に十分な110日分を取得
        hist = stock.history(period="6mo")

        if hist.empty:
            return {"error": "株価データなし", "ticker": ticker}

        close = hist["Close"]
        volume = hist["Volume"]

        # 移動平均
        ma5 = close.rolling(5).mean()
        ma25 = close.rolling(25).mean()
        ma75 = close.rolling(75).mean()

        def _last(s):
            v = s.iloc[-1]
            return round(float(v), 2) if not (isinstance(v, float) and math.isnan(v)) else None

        # 出来高移動平均比（直近出来高 / 20日平均出来高）
        vol_ma20 = volume.rolling(20).mean()
        vol_ratio = (
            round(float(volume.iloc[-1] / vol_ma20.iloc[-1]), 2)
            if vol_ma20.iloc[-1]
            else None
        )

        # 52週高値・安値
        year_data = stock.history(period="1y")
        week52_high = round(float(year_data["Close"].max()), 2) if not year_data.empty else None
        week52_low = round(float(year_data["Close"].min()), 2) if not year_data.empty else None

        # 直近20日の終値・出来高
        recent20 = hist.tail(20)
        prices_20d = [round(float(v), 2) for v in recent20["Close"].tolist()]
        volumes_20d = [int(v) for v in recent20["Volume"].tolist()]
        dates_20d = [d.strftime("%Y-%m-%d") for d in recent20.index]

        info = stock.fast_info
        current_price = round(float(info.last_price), 2) if hasattr(info, "last_price") else _last(close)

        return {
            "ticker": ticker,
            "current_price": current_price,
            "price_change_pct": round(
                float((close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] * 100), 2
            ) if len(close) >= 2 else None,
            "ma5": _last(ma5),
            "ma25": _last(ma25),
            "ma75": _last(ma75),
            "rsi14": _calc_rsi(close),
            "macd": _calc_macd(close),
            "volume_latest": int(volume.iloc[-1]),
            "volume_ma20_ratio": vol_ratio,
            "week52_high": week52_high,
            "week52_low": week52_low,
            "prices_20d": prices_20d,
            "volumes_20d": volumes_20d,
            "dates_20d": dates_20d,
        }

    except Exception as e:
        logger.error(f"yfinance取得エラー ({ticker}): {e}")
        return {"error": str(e), "ticker": ticker}


# ── RSS ────────────────────────────────────────────────────────────────────

# Yahoo!ファイナンス銘柄別ニュースRSSのURL
# 例: https://finance.yahoo.co.jp/rss/stocksNews?code=7203
YAHOO_FINANCE_RSS = "https://finance.yahoo.co.jp/rss/stocksNews?code={code}"
YAHOO_MARKET_RSS = "https://finance.yahoo.co.jp/rss/market"


def get_news_rss(ticker: str, max_items: int = 10) -> list[dict]:
    """
    Yahoo!ファイナンスRSSから銘柄の最新ニュースを取得する。
    ticker例: "7203.T" → code="7203"
    """
    code = ticker.replace(".T", "")
    url = YAHOO_FINANCE_RSS.format(code=code)

    try:
        feed = feedparser.parse(url)
        items = []
        for entry in feed.entries[:max_items]:
            items.append({
                "title": entry.get("title", ""),
                "link": entry.get("link", ""),
                "summary": entry.get("summary", ""),
                "published": entry.get("published", ""),
            })
        return items
    except Exception as e:
        logger.error(f"RSS取得エラー ({ticker}): {e}")
        return []


def get_market_news_rss(max_items: int = 15) -> list[dict]:
    """
    Yahoo!ファイナンス市場全体ニュースRSSを取得する。
    """
    try:
        feed = feedparser.parse(YAHOO_MARKET_RSS)
        items = []
        for entry in feed.entries[:max_items]:
            items.append({
                "title": entry.get("title", ""),
                "link": entry.get("link", ""),
                "summary": entry.get("summary", ""),
                "published": entry.get("published", ""),
            })
        return items
    except Exception as e:
        logger.error(f"市場ニュースRSS取得エラー: {e}")
        return []


# ── 統合収集エントリポイント ────────────────────────────────────────────────

def collect_all_data(tickers: list[str]) -> dict:
    """
    指定銘柄リストの全データを収集してまとめて返す。

    Args:
        tickers: ["7203.T", "6758.T"] 形式のリスト

    Returns:
        {
          "collected_at": "2025-03-08T08:00:00",
          "stocks": {
              "7203.T": {
                  "technical": {...},
                  "fundamental": {...},
                  "news": [...],
              },
              ...
          },
          "market_news": [...],
        }
    """
    result: dict = {
        "collected_at": datetime.now().isoformat(),
        "stocks": {},
        "market_news": [],
    }

    # 市場全体ニュース
    result["market_news"] = get_market_news_rss()

    for ticker in tickers:
        logger.info(f"データ収集中: {ticker}")
        technical = get_stock_data(ticker)
        fundamental = get_fundamental_data(ticker)
        news = get_news_rss(ticker)

        result["stocks"][ticker] = {
            "technical": technical,
            "fundamental": fundamental,
            "news": news,
        }

    return result
