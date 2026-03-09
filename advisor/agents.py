"""
マルチエージェント議論エンジン
Step1: 市場スクリーニング
Step2: 強気アナリスト
Step3: 弱気アナリスト
Step4: リスク管理官
Step5: モデレーター統合
"""

import json
import logging
import os
import re
import time
from datetime import date
from typing import Optional

import anthropic
from google import genai
from google.genai import types as gtypes

from advisor.data import collect_stock_data, get_market_news_rss
from advisor.prompts import (
    SCREENER_SYSTEM, SCREENER_USER_TEMPLATE,
    BULL_SYSTEM, BULL_USER_TEMPLATE,
    BEAR_SYSTEM, BEAR_USER_TEMPLATE,
    RISK_SYSTEM, RISK_USER_TEMPLATE,
    MODERATOR_SYSTEM, MODERATOR_USER_TEMPLATE,
    build_market_search_queries,
    build_stock_search_queries,
)

logger = logging.getLogger(__name__)

MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 4096


def _get_client() -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


def _get_gemini_client() -> genai.Client:
    return genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))


# ── web_search ───────────────────────────────────────────────────────────────

def _is_rate_limit(e: Exception) -> bool:
    return "429" in str(e) or "rate_limit" in str(e).lower()


def _api_call_with_retry(fn, label: str = ""):
    """任意のAPI呼び出しを429 exponential backoffでラップする"""
    max_retries = 6
    wait = 90  # 初回90秒待機（トークンウィンドウのリセットを待つ）
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            if not _is_rate_limit(e):
                raise
            if attempt == max_retries - 1:
                raise
            logger.warning(f"[{label}] 429 rate limit。{wait}秒後にリトライ ({attempt+1}/{max_retries})")
            time.sleep(wait)
            wait = int(wait * 1.5)


_FACT_ONLY_INSTRUCTION = (
    "以下のクエリで検索し、「確定した数値・公式発表・経済指標」のみを報告してください。\n"
    "【厳守ルール】\n"
    "- 報告するのは数値・日付・公式発表のみ。例: 『日経平均終値: 38,500円 (3/9)』\n"
    "- アナリスト意見・予測・推奨・目標株価・センチメントは一切不要\n"
    "- 『〜と見られる』『〜の可能性』『〜が期待される』などの推測表現は使わない\n"
    "- ニュース記事の論調や見出しの雰囲気は無視する\n"
    "- 数値が取得できない場合は『取得不可』とだけ記載する\n"
    "- 形式: 箇条書き（指標名: 数値 出典日時）\n\n"
    "クエリ: {query}"
)


def _gemini_search(queries: list[str]) -> str:
    """
    GeminiのGoogle検索グラウンディングでファクトのみを収集する。
    失敗した場合は空文字を返し、yfinanceデータのみで分析を続行する。
    """
    api_key = os.getenv("GOOGLE_API_KEY", "")
    if not api_key:
        logger.info("GOOGLE_API_KEY未設定。Gemini検索をスキップ。")
        return ""

    gclient = _get_gemini_client()
    results = []
    consecutive_failures = 0

    for query in queries:
        if consecutive_failures >= 3:
            logger.warning("Gemini連続失敗3回。残りクエリをスキップ。")
            break
        wait = 35
        for attempt in range(4):
            try:
                response = gclient.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=_FACT_ONLY_INSTRUCTION.format(query=query),
                    config=gtypes.GenerateContentConfig(
                        tools=[gtypes.Tool(google_search=gtypes.GoogleSearch())],
                        temperature=0.0,
                    ),
                )
                text = response.text or ""
                if text:
                    results.append(f"【{query}】\n{text}")
                consecutive_failures = 0
                logger.info(f"Gemini検索完了: {query[:30]}")
                break
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    if attempt < 3:
                        logger.warning(f"Gemini 429。{wait}秒後にリトライ ({attempt+1}/3)")
                        time.sleep(wait)
                        wait = int(wait * 1.5)
                    else:
                        consecutive_failures += 1
                        logger.warning(f"Gemini 429上限超過。スキップ: {query[:30]}")
                else:
                    consecutive_failures += 1
                    logger.warning(f"Gemini失敗 ({query[:30]}): {e}")
                    break

    return "\n\n".join(results)


# ── エージェント呼び出し ──────────────────────────────────────────────────────

def _call_agent(client: anthropic.Anthropic, system: str, user: str) -> str:
    def fn():
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return response.content[0].text

    return _api_call_with_retry(fn, "agent")


# ── ポートフォリオサマリー生成 ───────────────────────────────────────────────

def _portfolio_summary(portfolio: list[dict], stock_data: dict) -> str:
    if not portfolio:
        return "なし（初日）"
    lines = []
    for p in portfolio:
        ticker = p["ticker"]
        current = stock_data.get("stocks", {}).get(ticker, {}).get("technical", {}).get("current_price")
        avg = p["avg_price"]
        shares = p["shares"]
        if current:
            pnl_pct = round((current - avg) / avg * 100, 2)
            pnl_yen = int((current - avg) * shares)
            lines.append(
                f"- {ticker}({p.get('company_name', '')}) "
                f"{shares}株 @{avg:.0f}円 → 現在{current:.0f}円 "
                f"({'+' if pnl_pct >= 0 else ''}{pnl_pct}%, {'+' if pnl_yen >= 0 else ''}{pnl_yen:,}円)"
            )
        else:
            lines.append(f"- {ticker}({p.get('company_name', '')}) {shares}株 @{avg:.0f}円")
    return "\n".join(lines)


def _stock_value(portfolio: list[dict], stock_data: dict) -> int:
    total = 0
    for p in portfolio:
        ticker = p["ticker"]
        current = stock_data.get("stocks", {}).get(ticker, {}).get("technical", {}).get("current_price")
        price = current if current else p["avg_price"]
        total += int(price * p["shares"])
    return total


# ── 広域銘柄ユニバース（TSEプライム流動性上位〜500銘柄） ──────────────────
# 日経225 + TOPIX Large70 + 主要中型株をセクター横断で収録
# 量的事前スクリーニングでトップ50に絞り込み → AIスクリーナーが最終選定
BROAD_UNIVERSE = [
    # 自動車・輸送機器
    "7203.T", "7267.T", "7270.T", "7201.T", "7269.T", "7211.T", "7261.T",
    "6902.T", "7272.T", "7282.T", "7240.T", "7248.T", "7251.T", "7309.T",
    "7965.T", "6471.T", "6481.T",
    # 電機・半導体・精密機器
    "6758.T", "6861.T", "6367.T", "8035.T", "6723.T", "6857.T", "6594.T",
    "6501.T", "6503.T", "6752.T", "6971.T", "6976.T", "6981.T", "7735.T",
    "7741.T", "6645.T", "6326.T", "6301.T", "6305.T", "6473.T", "6479.T",
    "6504.T", "6506.T", "6586.T", "6674.T", "6702.T", "6724.T", "6728.T",
    "6740.T", "6754.T", "6762.T", "6770.T", "6807.T", "6841.T", "6869.T",
    "6954.T", "6963.T", "6966.T", "6967.T", "7731.T", "7733.T", "7762.T",
    "6472.T", "6417.T", "6383.T", "6315.T", "6273.T", "6268.T", "6383.T",
    # IT・ソフトウェア・通信
    "9984.T", "9432.T", "9433.T", "9434.T", "4755.T", "3659.T", "4689.T",
    "9613.T", "9719.T", "4307.T", "3668.T", "4726.T", "3769.T", "3923.T",
    "4171.T", "4543.T", "2432.T", "4185.T", "4783.T", "9749.T", "9783.T",
    "2371.T", "3092.T", "3697.T", "4004.T", "4021.T", "4386.T", "4443.T",
    "4565.T", "4587.T", "4776.T", "4911.T", "6027.T", "6088.T", "6098.T",
    "6178.T", "6254.T", "6532.T", "6539.T", "7065.T", "9107.T", "9143.T",
    "3558.T", "3962.T", "4051.T", "4053.T",
    # 金融・銀行・証券・保険
    "8306.T", "8316.T", "8411.T", "8309.T", "8604.T", "8601.T",
    "8697.T", "8750.T", "8725.T", "8630.T", "8766.T", "8795.T",
    "8253.T", "8354.T", "8355.T", "8359.T", "8361.T", "8385.T",
    "7186.T", "8418.T", "8473.T", "8593.T",
    # 小売・消費財
    "9983.T", "3382.T", "8267.T", "2651.T", "2784.T", "7459.T",
    "3088.T", "3099.T", "2668.T", "3289.T", "7453.T", "7532.T",
    "3086.T", "3197.T", "2730.T", "7516.T", "2782.T", "7513.T",
    "7412.T", "3048.T", "2670.T", "9843.T", "8905.T",
    # 素材・化学
    "4063.T", "4188.T", "4183.T", "4005.T", "4452.T", "5019.T", "5020.T",
    "5101.T", "5108.T", "4021.T", "4041.T", "4042.T", "4061.T", "4114.T",
    "4217.T", "4272.T", "4324.T", "4331.T", "4347.T", "4401.T", "4403.T",
    "4406.T", "4409.T", "4424.T", "4461.T", "4471.T", "4521.T", "4543.T",
    "4612.T", "4631.T", "4641.T", "4661.T", "4676.T", "4680.T", "4689.T",
    "4694.T", "4704.T", "4722.T", "4732.T", "4733.T", "4746.T", "4752.T",
    "4763.T", "4768.T", "4779.T", "4792.T",
    # 鉄鋼・非鉄金属
    "5401.T", "5411.T", "5713.T", "5802.T", "5706.T", "5423.T", "5440.T",
    "5463.T", "5702.T", "5727.T", "5741.T", "5743.T", "5803.T",
    # 不動産
    "8801.T", "8802.T", "8830.T", "3003.T", "8804.T", "8803.T",
    "3289.T", "3231.T", "3232.T", "3234.T", "3244.T", "3252.T",
    "3261.T", "3263.T", "3264.T", "3267.T", "3277.T", "8905.T",
    # 建設・エンジニアリング
    "1801.T", "1802.T", "1803.T", "1925.T", "1928.T", "1812.T", "1820.T",
    "1826.T", "1963.T", "5233.T", "5244.T", "1808.T", "1824.T",
    "1860.T", "1878.T", "1893.T", "1911.T", "1944.T", "1961.T", "1983.T",
    # 食品・飲料・タバコ
    "2502.T", "2503.T", "2587.T", "2269.T", "2201.T", "2914.T", "2802.T",
    "2282.T", "2264.T", "2593.T", "2211.T", "2212.T", "2213.T", "2221.T",
    "2229.T", "2264.T", "2270.T", "2281.T", "2292.T", "2296.T", "2531.T",
    "2579.T", "2594.T",
    # 製薬・ヘルスケア
    "4568.T", "4519.T", "4506.T", "4507.T", "4523.T", "4578.T", "4151.T",
    "4502.T", "4503.T", "4516.T", "4528.T", "4530.T", "4535.T", "4536.T",
    "4540.T", "4544.T", "4549.T", "4550.T", "4554.T", "4556.T", "4563.T",
    "4564.T", "4569.T", "4571.T", "4574.T", "4576.T",
    # ゲーム・エンタメ・メディア
    "7974.T", "9766.T", "9697.T", "2432.T", "9684.T", "3632.T", "3765.T",
    "4344.T", "4755.T", "3735.T", "2137.T", "2378.T", "2464.T", "4751.T",
    # 商社
    "8058.T", "8053.T", "8031.T", "8001.T", "8002.T", "8015.T",
    "8025.T", "8030.T", "8035.T", "8043.T", "8088.T", "9506.T",
    # エネルギー・資源
    "1605.T", "5019.T", "5020.T", "9531.T", "9532.T", "9502.T", "9503.T",
    "9504.T", "9505.T",
    # 物流・運輸
    "9104.T", "9101.T", "9107.T", "9143.T", "9064.T", "9147.T",
    "9022.T", "9020.T", "9021.T", "9048.T",
    # その他製造・機械
    "6201.T", "6301.T", "6305.T", "6361.T", "6366.T", "6370.T",
    "6374.T", "6378.T", "6381.T", "6382.T", "6391.T", "6395.T",
    "6399.T", "6402.T", "6407.T", "6412.T", "6413.T",
]
# 重複除去
BROAD_UNIVERSE = list(dict.fromkeys(BROAD_UNIVERSE))

# フォールバック（全銘柄取得失敗時の最小セット）
FALLBACK_UNIVERSE = [
    "7203.T", "6758.T", "9984.T", "6861.T", "8306.T",
    "9432.T", "6367.T", "7974.T", "4063.T", "8035.T",
]


# ── 広域ユニバースの高速事前スクリーニング ────────────────────────────────────

def _fast_prescreen(universe: list[str], top_n: int = 50) -> tuple[list[str], str]:
    """
    全ユニバースをバッチダウンロードで高速スクリーニング。
    1日変化率・出来高比でスコアリングしてトップN銘柄を返す。
    戻り値: (上位ティッカーリスト, スクリーナー向けサマリーテキスト)
    """
    import yfinance as yf
    import pandas as pd
    import numpy as np

    logger.info(f"高速事前スクリーニング開始: {len(universe)}銘柄")
    try:
        data = yf.download(
            universe,
            period="25d",
            auto_adjust=True,
            progress=False,
            group_by="ticker",
        )
        if data.empty:
            logger.warning("yfinance一括DL: データなし。フォールバック使用。")
            return universe[:top_n], "データ取得失敗"

        results = []
        for ticker in universe:
            try:
                if isinstance(data.columns, pd.MultiIndex):
                    if ticker not in data.columns.get_level_values(0):
                        continue
                    t_data = data[ticker]
                else:
                    t_data = data

                close = t_data["Close"].dropna()
                volume = t_data["Volume"].dropna()

                if len(close) < 6:
                    continue

                cur = float(close.iloc[-1])
                prev = float(close.iloc[-2])
                prev5 = float(close.iloc[-6]) if len(close) >= 6 else prev
                vol_today = float(volume.iloc[-1])
                vol_avg = float(volume.iloc[:-1].mean()) if len(volume) > 1 else vol_today

                if cur <= 0 or prev <= 0 or vol_avg <= 0:
                    continue

                chg1d = (cur - prev) / prev * 100
                chg5d = (cur - prev5) / prev5 * 100
                vol_ratio = vol_today / vol_avg

                # スコア: 価格変動幅 × 出来高急増 （絶対値で注目度を測る）
                score = (abs(chg1d) * 2 + abs(chg5d)) * (vol_ratio ** 0.5)

                results.append({
                    "ticker": ticker,
                    "price": round(cur, 0),
                    "chg1d": round(chg1d, 2),
                    "chg5d": round(chg5d, 2),
                    "vol_ratio": round(vol_ratio, 2),
                    "score": round(score, 3),
                })
            except Exception:
                continue

        if not results:
            logger.warning("スコアリング結果なし。フォールバック使用。")
            return universe[:top_n], "スコアリング失敗"

        results.sort(key=lambda x: x["score"], reverse=True)
        top = results[:top_n]
        top_tickers = [r["ticker"] for r in top]

        # スクリーナー向けサマリー（コンパクト表形式）
        lines = [
            f"## 全{len(universe)}銘柄スクリーニング → 注目上位{len(top)}銘柄",
            "| ティッカー | 株価 | 1日変化 | 5日変化 | 出来高比 | スコア |",
            "|---|---|---|---|---|---|",
        ]
        for r in top:
            lines.append(
                f"| {r['ticker']} | {r['price']:,.0f}円 | "
                f"{'+' if r['chg1d'] >= 0 else ''}{r['chg1d']}% | "
                f"{'+' if r['chg5d'] >= 0 else ''}{r['chg5d']}% | "
                f"{r['vol_ratio']}x | {r['score']} |"
            )

        logger.info(f"高速事前スクリーニング完了: {len(results)}銘柄スコアリング → 上位{len(top)}銘柄抽出")
        return top_tickers, "\n".join(lines)

    except Exception as e:
        logger.warning(f"高速事前スクリーニング例外: {e}")
        return universe[:top_n], f"スクリーニング例外: {e}"

# ── 候補銘柄をスクリーニング結果からパース ─────────────────────────────────

def _parse_candidates(screening_text: str) -> list[str]:
    """スクリーニング結果からティッカーコードを抽出する"""
    pattern = r'\b(\d{4})\.T\b'
    tickers = re.findall(pattern, screening_text)
    result = [f"{t}.T" for t in dict.fromkeys(tickers)]
    # スクリーナーが銘柄を出せなかった場合はフォールバックユニバースを使う
    if not result:
        logger.warning("スクリーナーが銘柄を返さなかった。フォールバックユニバースを使用。")
        result = FALLBACK_UNIVERSE[:5]
    return result


# ── エージェント向けデータサマリー ─────────────────────────────────────────────

def _format_data_for_agents(raw_data: dict) -> str:
    """raw_dataをトークン効率の良いテキストに変換する"""
    lines = []
    for ticker, stock in raw_data.get("stocks", {}).items():
        tech = stock.get("technical", {})
        fund = stock.get("fundamental", {})
        news = stock.get("news", [])
        web  = stock.get("web_search", "")

        lines.append(f"### {ticker}")
        lines.append(f"株価: {tech.get('current_price')}円 (前日比: {tech.get('price_change_pct')}%)")
        lines.append(f"MA5/25/75: {tech.get('ma5')}/{tech.get('ma25')}/{tech.get('ma75')}")
        lines.append(f"RSI14: {tech.get('rsi14')} / MACD: {tech.get('macd', {}).get('macd')} Signal: {tech.get('macd', {}).get('signal')}")
        lines.append(f"出来高MA比: {tech.get('volume_ma20_ratio')} / 52週高値: {tech.get('week52_high')} 安値: {tech.get('week52_low')}")

        if fund and not fund.get("error"):
            lines.append(f"売上: {fund.get('net_sales')} / 営業利益: {fund.get('operating_profit')} / PER: {fund.get('per')} / PBR: {fund.get('pbr')}")

        if news:
            lines.append("直近ニュース:")
            for n in news[:3]:
                lines.append(f"  - {n.get('title', '')} ({n.get('published', '')[:16]})")

        if web:
            # web_searchは先頭500文字のみ
            lines.append(f"web検索: {web[:500]}")

        lines.append("")
    return "\n".join(lines)


# ── 自信度パース ──────────────────────────────────────────────────────────────

def _extract_confidence(text: str) -> int:
    m = re.search(r"自信度.*?(\d{1,3})\s*%", text)
    return int(m.group(1)) if m else 50


# ── メインエントリポイント ────────────────────────────────────────────────────

def _format_history(history: list[dict]) -> str:
    """
    過去の提案・取引結果を「学習用レビュー」フォーマットに変換する。
    直近5件を新しい順に表示。
    """
    if not history:
        return "なし（初日）"

    lines = []
    for h in history[:5]:
        lines.append(f"### {h['date']}")

        # 提案の要点（最終提案の最初の200字）
        proposal_preview = (h.get("final_proposal") or "")[:200]
        lines.append(f"提案要点: {proposal_preview}...")

        # 実際のトレード
        trades = h.get("trades", [])
        if trades:
            for t in trades:
                action_jp = {"buy": "買い", "sell": "売り", "hold": "見送り"}.get(t.get("action", ""), t.get("action", ""))
                pnl = t.get("pnl")
                pnl_str = f" 損益: {'+' if pnl and pnl >= 0 else ''}{pnl:,}円" if pnl is not None else ""
                lines.append(
                    f"  実行: {t.get('ticker')} {action_jp} "
                    f"{t.get('shares', 0)}株 @{t.get('price', 0):,.0f}円{pnl_str}"
                )
        else:
            lines.append("  実行: 未記録")

        # 自信度
        conf = h.get("confidence")
        if conf:
            lines.append(f"  自信度: {conf}%")

        lines.append("")
    return "\n".join(lines)


def generate_proposal(settings: dict, portfolio: list[dict], history: list[dict] = None) -> dict:
    """
    マルチエージェント議論を実行し最終提案を返す。

    Args:
        settings: {capital, current_cash, target_amount, deadline}
        portfolio: [{ticker, company_name, shares, avg_price}, ...]

    Returns:
        {screening_result, bull_analysis, bear_analysis, risk_analysis,
         final_proposal, confidence, raw_data}
    """
    client = _get_client()
    today_str = date.today().isoformat()
    history = history or []
    history_text = _format_history(history)

    remaining_days = (date.fromisoformat(str(settings["deadline"])) - date.today()).days
    current_cash = settings["current_cash"]
    target_amount = settings["target_amount"]
    capital = settings["capital"]

    # ── Step 1a: 広域ユニバース高速事前スクリーニング ─────────────────────────
    logger.info(f"Step1a: 広域スクリーニング開始（{len(BROAD_UNIVERSE)}銘柄）")
    held_tickers = [p["ticker"] for p in portfolio]

    # 高速スクリーニング: 全ユニバースから注目上位50銘柄を抽出
    top_tickers, prescreen_summary = _fast_prescreen(BROAD_UNIVERSE, top_n=50)

    # 保有銘柄は必ず含める（売り候補として分析が必要）
    prefetch_tickers = list(dict.fromkeys(top_tickers + held_tickers))

    # 上位銘柄のみ深い分析（MA/RSI/MACDなど）
    raw_data = collect_stock_data(prefetch_tickers)
    fallback_summary = _format_data_for_agents(raw_data)

    # ── Step 1b: Gemini Google検索で市場情報を網羅的に収集 ───────────────────
    logger.info("Step1b: Gemini検索（市場情報・銘柄情報）")
    market_queries = build_market_search_queries()

    # 市場全体クエリのみ（銘柄個別はyfinanceで代替、クエリ数を最小化）
    market_info = _gemini_search(market_queries)

    # RSS市場ニュース（全文）
    rss_news = get_market_news_rss(max_items=15)
    rss_text = "\n".join(f"- {n['title']}: {n['summary']}" for n in rss_news)

    sections = []
    if market_info:
        sections.append(f"## Gemini Google検索結果\n{market_info}")
    if rss_text.strip():
        sections.append(f"## 市場ニュース（RSS）\n{rss_text}")
    # 事前スクリーニング結果（全ユニバース中の注目銘柄ランキング）
    sections.append(f"{prescreen_summary}")
    sections.append(f"## 上位候補の詳細データ（株価・テクニカル）\n{fallback_summary}")
    full_market_info = "\n\n".join(sections)

    # Gemini検索はClaudeのトークン制限に影響しないので待機不要
    logger.info("Step1b完了。65秒待機（Claudeスクリーナー前）...")

    # ── Step 1c: スクリーナーエージェント ────────────────────────────────────
    logger.info("Step1c: スクリーナー実行")
    portfolio_summary_text = _portfolio_summary(portfolio, raw_data)

    screener_user = SCREENER_USER_TEMPLATE.format(
        available_cash=current_cash,
        target_amount=target_amount,
        deadline=settings["deadline"],
        remaining_days=remaining_days,
        portfolio_summary=portfolio_summary_text,
        market_info=full_market_info,
        history=history_text,
    )
    screening_result = _call_agent(client, SCREENER_SYSTEM, screener_user)

    logger.info("スクリーナー完了。30秒待機...")
    time.sleep(30)

    # 候補銘柄を抽出（フォールバック込み）
    candidate_tickers = _parse_candidates(screening_result)
    all_tickers = list(dict.fromkeys(candidate_tickers + held_tickers))

    logger.info(f"候補銘柄: {all_tickers}")

    # 未収集の銘柄があれば追加収集
    new_tickers = [t for t in all_tickers if t not in raw_data["stocks"]]
    if new_tickers:
        extra = collect_stock_data(new_tickers)
        raw_data["stocks"].update(extra["stocks"])

    # ポートフォリオサマリー（株価データ込み）
    portfolio_summary_text = _portfolio_summary(portfolio, raw_data)
    sv = _stock_value(portfolio, raw_data)
    total_assets = current_cash + sv

    candidates_text = screening_result
    context_text = (
        f"現在の現金: {current_cash:,}円\n"
        f"保有株式評価額: {sv:,}円\n"
        f"合計資産: {total_assets:,}円\n"
        f"目標金額: {target_amount:,}円\n"
        f"期日: {settings['deadline']}（残り{remaining_days}日）\n"
        f"今日: {today_str}"
    )
    # 候補銘柄の完全データ（テキスト形式）をエージェントに渡す
    data_text = _format_data_for_agents(raw_data)

    # ── Step 2: 強気アナリスト ────────────────────────────────────────────────
    logger.info("Step2: 強気アナリスト")
    bull_analysis = _call_agent(
        client, BULL_SYSTEM,
        BULL_USER_TEMPLATE.format(
            candidates=candidates_text,
            context=context_text,
            data=data_text,
        )
    )

    logger.info("Step2完了。30秒待機...")
    time.sleep(30)

    # ── Step 3: 弱気アナリスト ────────────────────────────────────────────────
    logger.info("Step3: 弱気アナリスト")
    bear_analysis = _call_agent(
        client, BEAR_SYSTEM,
        BEAR_USER_TEMPLATE.format(
            candidates=candidates_text,
            context=context_text,
            data=data_text,
        )
    )

    logger.info("Step3完了。30秒待機...")
    time.sleep(30)

    # ── Step 4: リスク管理官 ──────────────────────────────────────────────────
    logger.info("Step4: リスク管理官")
    risk_analysis = _call_agent(
        client, RISK_SYSTEM,
        RISK_USER_TEMPLATE.format(
            capital=capital,
            current_cash=current_cash,
            stock_value=sv,
            total_assets=total_assets,
            target_amount=target_amount,
            deadline=settings["deadline"],
            remaining_days=remaining_days,
            candidates=candidates_text,
            portfolio_summary=portfolio_summary_text,
        )
    )

    logger.info("Step4完了。30秒待機...")
    time.sleep(30)

    # ── Step 5: モデレーター ──────────────────────────────────────────────────
    logger.info("Step5: モデレーター統合")
    final_proposal = _call_agent(
        client, MODERATOR_SYSTEM,
        MODERATOR_USER_TEMPLATE.format(
            current_cash=current_cash,
            target_amount=target_amount,
            deadline=settings["deadline"],
            portfolio_summary=portfolio_summary_text,
            screening_result=screening_result,
            bull_analysis=bull_analysis,
            bear_analysis=bear_analysis,
            risk_analysis=risk_analysis,
            history=history_text,
        )
    )

    confidence = _extract_confidence(final_proposal)
    logger.info(f"提案生成完了 (自信度: {confidence}%)")

    return {
        "raw_data": raw_data,
        "screening_result": screening_result,
        "bull_analysis": bull_analysis,
        "bear_analysis": bear_analysis,
        "risk_analysis": risk_analysis,
        "final_proposal": final_proposal,
        "confidence": confidence,
    }
