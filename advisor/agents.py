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


def _gemini_search(queries: list[str]) -> str:
    """
    GeminiのGoogle検索グラウンディングで複数クエリを検索する。
    Claudeのweb_searchと違いトークン制限がなく、レート制限も緩い。
    """
    gclient = _get_gemini_client()
    results = []

    for query in queries:
        try:
            response = gclient.models.generate_content(
                model="gemini-2.0-flash",
                contents=f"以下について検索し、投資判断に役立つ最新情報・数値・事実を詳しく答えてください。\n\n{query}",
                config=gtypes.GenerateContentConfig(
                    tools=[gtypes.Tool(google_search=gtypes.GoogleSearch())],
                    temperature=0.1,
                ),
            )
            text = response.text or "（結果なし）"
            results.append(f"【{query}】\n{text}")
            logger.info(f"Gemini検索完了: {query[:30]}")
        except Exception as e:
            logger.warning(f"Gemini検索失敗 ({query}): {e}")
            results.append(f"【{query}】\n（取得失敗: {e}）")

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


# ── フォールバック銘柄ユニバース ──────────────────────────────────────────────
# web_searchが失敗してもyfinanceで必ずデータ収集できる流動性の高い銘柄
FALLBACK_UNIVERSE = [
    "7203.T",  # トヨタ自動車
    "6758.T",  # ソニーグループ
    "9984.T",  # ソフトバンクグループ
    "6861.T",  # キーエンス
    "8306.T",  # 三菱UFJフィナンシャル
    "9432.T",  # 日本電信電話
    "6367.T",  # ダイキン工業
    "7974.T",  # 任天堂
    "4063.T",  # 信越化学工業
    "8035.T",  # 東京エレクトロン
]

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

def generate_proposal(settings: dict, portfolio: list[dict]) -> dict:
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

    remaining_days = (date.fromisoformat(str(settings["deadline"])) - date.today()).days
    current_cash = settings["current_cash"]
    target_amount = settings["target_amount"]
    capital = settings["capital"]

    # ── Step 1a: フォールバックユニバース + 保有銘柄のyfinance/RSSデータを収集 ──
    logger.info("Step1a: 株価・ニュースデータ収集")
    held_tickers = [p["ticker"] for p in portfolio]
    prefetch_tickers = list(dict.fromkeys(FALLBACK_UNIVERSE + held_tickers))
    raw_data = collect_stock_data(prefetch_tickers)
    fallback_summary = _format_data_for_agents(raw_data)

    # ── Step 1b: Gemini Google検索で市場情報を網羅的に収集 ───────────────────
    logger.info("Step1b: Gemini検索（市場情報・銘柄情報）")
    market_queries = build_market_search_queries()

    # 候補銘柄の個別検索クエリも追加
    stock_queries = []
    for ticker in FALLBACK_UNIVERSE[:5]:
        code = ticker.replace(".T", "")
        stock_queries.extend(build_stock_search_queries(code, code))

    all_queries = market_queries + stock_queries
    market_info = _gemini_search(all_queries)

    # RSS市場ニュース（全文）
    rss_news = get_market_news_rss(max_items=15)
    rss_text = "\n".join(f"- {n['title']}: {n['summary']}" for n in rss_news)

    full_market_info = (
        f"## Gemini Google検索結果\n{market_info}\n\n"
        f"## 市場ニュース（RSS）\n{rss_text}\n\n"
        f"## 主要銘柄の株価・テクニカルデータ\n{fallback_summary}"
    )

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
