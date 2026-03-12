"""
データ収集モジュール
- yfinance: 株価・テクニカル指標
- J-Quants API: 決算・ファンダメンタル
- Yahoo!ファイナンスRSS: 最新ニュース
"""

import io
import math
import logging
import os
import requests
import feedparser
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

JQUANTS_BASE = "https://api.jquants.com/v1"

# ── JPX公開データ ─────────────────────────────────────────────────────────────

_JPX_URL = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls"
_JPX_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; kabu-advisor/1.0)"}

def get_all_tse_tickers_from_jpx(
    markets: list[str] | None = None,
    size_filter: list[str] | None = None,
) -> list[str]:
    """
    JPX公開Excelから全TSE内国株ティッカーリストを取得する。
    markets: None=全内国株, ["プライム"] 等で市場フィルタ
    size_filter: ["TOPIX Core30","TOPIX Large70","TOPIX Mid400"] 等で時価総額フィルタ
    """
    try:
        resp = requests.get(_JPX_URL, headers=_JPX_HEADERS, timeout=30)
        resp.raise_for_status()
        df = pd.read_excel(io.BytesIO(resp.content), dtype=str)

        code_col = next(
            (c for c in df.columns if "コード" in str(c) or str(c).lower() in ("code",)),
            df.columns[0],
        )
        market_col = next((c for c in df.columns if "市場" in str(c)), None)
        size_col = next((c for c in df.columns if str(c) == "規模区分"), None)

        tickers = []
        for _, row in df.iterrows():
            raw_code = str(row[code_col]).strip().split(".")[0]
            if len(raw_code) != 4 or not raw_code.isdigit():
                continue

            market_val = str(row.get(market_col, "")) if market_col else ""
            if market_col and "内国株式" not in market_val:
                continue
            if markets and not any(m in market_val for m in markets):
                continue

            if size_filter and size_col:
                size_val = str(row.get(size_col, ""))
                if not any(s in size_val for s in size_filter):
                    continue

            tickers.append(f"{raw_code}.T")

        logger.info(f"JPX: {len(tickers)}銘柄取得 (markets={markets}, size={size_filter})")
        return tickers

    except Exception as e:
        logger.warning(f"JPX Excelデータ取得失敗: {e}")
        return []


# ── J-Quants ────────────────────────────────────────────────────────────────

def get_all_tse_prime_tickers() -> list[str]:
    """
    J-Quants /v1/listed/info からTSEプライム全銘柄のティッカーリストを取得する。
    失敗した場合は空リストを返す（フォールバックに委ねる）。
    MarketCode: "0111"=TSEプライム, "0121"=TSEスタンダード, "0131"=TSEグロース
    """
    id_token = _jquants_id_token()
    if not id_token:
        return []
    try:
        headers = {"Authorization": f"Bearer {id_token}"}
        res = requests.get(
            f"{JQUANTS_BASE}/listed/info",
            headers=headers,
            timeout=30,
        )
        res.raise_for_status()
        raw = res.json()
        companies = raw.get("info", [])
        if not companies or not isinstance(companies, list):
            logger.warning(f"J-Quants listed/info: 予期しないレスポンス形式 keys={list(raw.keys())}")
            return []
        tickers = []
        for c in companies:
            code = c.get("Code", "")
            market = c.get("MarketCode", "") or c.get("Market", "") or c.get("market_code", "")
            # TSEプライム判定: MarketCode "0111" または "Prime"を含む文字列
            is_prime = (market == "0111") or ("prime" in str(market).lower()) or ("プライム" in str(market))
            # 普通株判定: コード末尾0
            is_common = len(code) >= 4 and code.endswith("0")
            if is_prime and is_common:
                tickers.append(f"{code[:4]}.T")
        logger.info(f"J-Quants: TSEプライム {len(tickers)}銘柄取得")
        return tickers
    except Exception as e:
        logger.warning(f"J-Quants listed/info 取得失敗: {e}")
        return []


def _jquants_id_token() -> Optional[str]:
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


def _get_fundamental_from_yfinance(ticker: str) -> dict:
    """
    yfinance .info からファンダメンタルデータを取得する（J-Quantsのフォールバック）。
    取得項目: PER・PBR・配当利回り・売上・営業利益・純利益・時価総額。
    """
    try:
        info = yf.Ticker(ticker).info
        if not info or info.get("trailingPE") is None and info.get("marketCap") is None:
            return {"error": "yfinance info取得失敗", "ticker": ticker, "source": "yfinance"}

        def _safe(key, default=None):
            v = info.get(key)
            return v if v is not None else default

        revenue = _safe("totalRevenue")
        op_income = _safe("operatingCashflow") or _safe("ebitda")  # 営業利益の近似
        net_income = _safe("netIncomeToCommon")
        market_cap = _safe("marketCap")

        return {
            "ticker": ticker,
            "source": "yfinance",
            "per": _safe("trailingPE"),
            "forward_per": _safe("forwardPE"),
            "pbr": _safe("priceToBook"),
            "dividend_yield": _safe("dividendYield"),  # yfinanceは0.027形式(=2.7%)で返す
            "net_sales": revenue,
            "operating_profit": op_income,
            "net_income": net_income,
            "market_cap": market_cap,
            "roe": _safe("returnOnEquity"),
            "roa": _safe("returnOnAssets"),
            "debt_to_equity": _safe("debtToEquity"),
            "current_ratio": _safe("currentRatio"),
            # J-Quants互換フィールド（なければNone）
            "fiscal_year": None,
            "sales_deviation_pct": None,
            "op_profit_deviation_pct": None,
        }
    except Exception as e:
        logger.warning(f"yfinance fundamental取得失敗 ({ticker}): {e}")
        return {"error": str(e), "ticker": ticker, "source": "yfinance"}


def get_fundamental_data(ticker: str) -> dict:
    code = ticker.replace(".T", "")
    id_token = _jquants_id_token()
    if not id_token:
        # J-Quants未設定 → yfinanceにフォールバック
        logger.debug(f"J-Quants未設定のためyfinanceでファンダメンタル取得: {ticker}")
        return _get_fundamental_from_yfinance(ticker)
    headers = {"Authorization": f"Bearer {id_token}"}
    try:
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
        net_sales = float(latest.get("NetSales") or 0)
        forecast_sales = float(latest.get("ForecastNetSales") or 0)
        op_profit = float(latest.get("OperatingProfit") or 0)
        forecast_op = float(latest.get("ForecastOperatingProfit") or 0)
        return {
            "ticker": ticker,
            "source": "jquants",
            "fiscal_year": latest.get("FiscalYear"),
            "net_sales": net_sales,
            "operating_profit": op_profit,
            "net_income": float(latest.get("Profit") or 0),
            "sales_deviation_pct": round((net_sales - forecast_sales) / forecast_sales * 100, 2) if forecast_sales else None,
            "op_profit_deviation_pct": round((op_profit - forecast_op) / forecast_op * 100, 2) if forecast_op else None,
            "per": latest.get("PriceEarningsRatio"),
            "pbr": latest.get("PriceBookValueRatio"),
            "dividend_yield": latest.get("DividendYield"),
        }
    except Exception as e:
        logger.error(f"J-Quants取得エラー ({ticker}): {e}")
        return {"error": str(e), "ticker": ticker}


# ── yfinance ────────────────────────────────────────────────────────────────

def _calc_rsi(series: pd.Series, period: int = 14) -> Optional[float]:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    val = rsi.iloc[-1]
    return round(float(val), 2) if not math.isnan(float(val)) else None


def _calc_macd(series: pd.Series) -> dict:
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal = macd_line.ewm(span=9, adjust=False).mean()
    hist = macd_line - signal
    def _v(s):
        v = s.iloc[-1]
        return round(float(v), 4) if not math.isnan(float(v)) else None
    return {"macd": _v(macd_line), "signal": _v(signal), "histogram": _v(hist)}


def _extract_ohlcv(batch_data: pd.DataFrame, ticker: str) -> tuple[pd.Series, pd.Series] | None:
    """
    yf.download() のバッチ結果から指定ティッカーの (close, volume) を取得する。
    yfinance バージョンによってMultiIndexの列順が異なるため両方に対応:
    - group_by="ticker" 形式: (ticker, price_type) → data[ticker]["Close"]
    - デフォルト形式:          (price_type, ticker) → data["Close"][ticker]
    - 単一銘柄形式:             data["Close"]
    """
    if batch_data is None or batch_data.empty:
        return None

    try:
        if not isinstance(batch_data.columns, pd.MultiIndex):
            # 単一銘柄の場合
            close = batch_data["Close"].dropna()
            volume = batch_data["Volume"].dropna()
            if not close.empty:
                return close, volume
            return None

        level0 = batch_data.columns.get_level_values(0).unique().tolist()
        level1 = batch_data.columns.get_level_values(1).unique().tolist()

        if ticker in level0:
            # (ticker, price_type) 形式
            t_df = batch_data[ticker]
            close = t_df["Close"].dropna()
            volume = t_df["Volume"].dropna()
        elif ticker in level1:
            # (price_type, ticker) 形式
            close = batch_data["Close"][ticker].dropna()
            volume = batch_data["Volume"][ticker].dropna()
        else:
            return None

        if close.empty or len(close) < 2:
            return None
        return close, volume

    except Exception as e:
        logger.debug(f"_extract_ohlcv ({ticker}): {e}")
        return None


def _technical_from_df(ticker: str, close: pd.Series, volume: pd.Series) -> dict:
    """close/volumeのSeriesからテクニカル指標を計算する"""
    def _last(s):
        v = s.iloc[-1]
        return round(float(v), 2) if not math.isnan(float(v)) else None

    ma5 = close.rolling(5).mean()
    ma25 = close.rolling(25).mean()
    ma75 = close.rolling(75).mean()
    vol_ma20 = volume.rolling(20).mean()
    _vm = vol_ma20.iloc[-1]
    vol_ratio = round(float(volume.iloc[-1] / _vm), 2) if (pd.notna(_vm) and _vm > 0) else None
    week52_high = round(float(close.max()), 2)
    week52_low = round(float(close.min()), 2)
    price_chg = round(float((close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] * 100), 2) if len(close) >= 2 else None

    return {
        "ticker": ticker,
        "current_price": _last(close),
        "price_change_pct": price_chg,
        "ma5": _last(ma5),
        "ma25": _last(ma25),
        "ma75": _last(ma75),
        "rsi14": _calc_rsi(close),
        "macd": _calc_macd(close),
        "volume_latest": int(volume.iloc[-1]),
        "volume_ma20_ratio": vol_ratio,
        "week52_high": week52_high,
        "week52_low": week52_low,
    }


# ── RSS ─────────────────────────────────────────────────────────────────────

def get_news_rss(ticker: str, max_items: int = 8) -> list[dict]:
    code = ticker.replace(".T", "")
    url = f"https://finance.yahoo.co.jp/rss/stocksNews?code={code}"
    try:
        feed = feedparser.parse(url)
        return [
            {"title": e.get("title", ""), "summary": e.get("summary", ""), "published": e.get("published", "")}
            for e in feed.entries[:max_items]
        ]
    except Exception as e:
        logger.error(f"RSS取得エラー ({ticker}): {e}")
        return []


def get_market_news_rss(max_items: int = 15) -> list[dict]:
    try:
        feed = feedparser.parse("https://finance.yahoo.co.jp/rss/market")
        return [
            {"title": e.get("title", ""), "summary": e.get("summary", ""), "published": e.get("published", "")}
            for e in feed.entries[:max_items]
        ]
    except Exception as e:
        logger.error(f"市場ニュースRSS取得エラー: {e}")
        return []


# ── 複数銘柄データ収集 ───────────────────────────────────────────────────────

def collect_stock_data(tickers: list[str]) -> dict:
    """
    指定銘柄の株価・テクニカル指標・ファンダメンタル・ニュースを収集。
    yf.download() バッチAPIを使用（yf.Ticker().history()より安定）。
    """
    result = {"collected_at": datetime.now().isoformat(), "stocks": {}}
    if not tickers:
        return result

    # ── バッチ株価ダウンロード（1年分）────────────────────────────────────────
    batch_data = None
    try:
        batch_data = yf.download(
            tickers,
            period="1y",
            auto_adjust=True,
            progress=False,
            group_by="ticker",
        )
        logger.info(f"バッチDL完了: {len(tickers)}銘柄")
    except Exception as e:
        logger.error(f"バッチDL失敗: {e}")

    # ── 銘柄ごとにテクニカル計算 + ファンダメンタル + ニュース ────────────────
    for ticker in tickers:
        technical = {"error": "データ取得失敗", "ticker": ticker}
        try:
            ohlcv = _extract_ohlcv(batch_data, ticker)
            if ohlcv is not None:
                close, volume = ohlcv
                technical = _technical_from_df(ticker, close, volume)
        except Exception as e:
            logger.error(f"テクニカル計算エラー ({ticker}): {e}")
            technical = {"error": str(e), "ticker": ticker}

        # 社名をyfinanceから取得（AIが記憶から誤った社名を出力しないよう正確な名前を渡す）
        company_name = ""
        try:
            info = yf.Ticker(ticker).fast_info
            company_name = getattr(info, "company_name", "") or ""
            if not company_name:
                info2 = yf.Ticker(ticker).info
                company_name = info2.get("shortName") or info2.get("longName") or ""
        except Exception:
            pass

        result["stocks"][ticker] = {
            "company_name": company_name,
            "technical": technical,
            "fundamental": get_fundamental_data(ticker),
            "news": get_news_rss(ticker),
        }

    success = sum(1 for v in result["stocks"].values() if "error" not in v.get("technical", {}) and v["technical"].get("current_price") is not None)
    logger.info(f"データ収集完了: {success}/{len(tickers)}銘柄 成功")
    return result
