"""
マルチエージェント議論エンジン
Step2〜5: bull → bear → risk → moderator の順で分析・統合する
"""

import json
import logging
import os
import re
from datetime import date, datetime
from typing import Optional

import anthropic

from advisor.data import collect_all_data
from advisor.prompts import (
    BULL_SYSTEM, BULL_USER_TEMPLATE,
    BEAR_SYSTEM, BEAR_USER_TEMPLATE,
    RISK_SYSTEM, RISK_USER_TEMPLATE,
    MODERATOR_SYSTEM, MODERATOR_USER_TEMPLATE,
    build_search_queries,
)

logger = logging.getLogger(__name__)

MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 4096


def _get_client() -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


# ── web_search でデータを補強 ────────────────────────────────────────────────

def _web_search_supplement(
    client: anthropic.Anthropic,
    queries: list[str],
) -> str:
    """
    Claude の web_search ツールを使って検索結果を収集し、テキストにまとめる。
    """
    tool_def = {
        "type": "web_search_20250305",
        "name": "web_search",
        "max_uses": len(queries),
    }

    # クエリをまとめて1回のAPI呼び出しで検索させる
    query_list = "\n".join(f"- {q}" for q in queries)
    prompt = f"""以下のクエリで順番にウェブ検索を行い、各結果の要点をまとめてください。
投資判断に役立つ数値・事実・最新情報を優先してください。

{query_list}
"""
    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=3000,
            tools=[tool_def],
            messages=[{"role": "user", "content": prompt}],
        )
        # テキストブロックを結合
        texts = [b.text for b in response.content if hasattr(b, "text")]
        return "\n\n".join(texts)
    except Exception as e:
        logger.warning(f"web_search失敗: {e}")
        return f"（web_search失敗: {e}）"


# ── 各エージェント呼び出し ──────────────────────────────────────────────────

def _call_agent(
    client: anthropic.Anthropic,
    system: str,
    user: str,
) -> str:
    """シンプルなエージェント呼び出し"""
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return response.content[0].text


def run_bull_agent(client: anthropic.Anthropic, context: str, data: str) -> str:
    logger.info("強気アナリスト分析開始")
    user = BULL_USER_TEMPLATE.format(context=context, data=data)
    return _call_agent(client, BULL_SYSTEM, user)


def run_bear_agent(client: anthropic.Anthropic, context: str, data: str) -> str:
    logger.info("弱気アナリスト分析開始")
    user = BEAR_USER_TEMPLATE.format(context=context, data=data)
    return _call_agent(client, BEAR_SYSTEM, user)


def run_risk_agent(
    client: anthropic.Anthropic,
    capital: int,
    current_amount: int,
    target_amount: int,
    deadline: str,
    today: str,
    data_summary: str,
) -> str:
    logger.info("リスク管理官分析開始")
    user = RISK_USER_TEMPLATE.format(
        capital=capital,
        current_amount=current_amount,
        target_amount=target_amount,
        deadline=deadline,
        today=today,
        data_summary=data_summary,
    )
    return _call_agent(client, RISK_SYSTEM, user)


def run_moderator_agent(
    client: anthropic.Anthropic,
    current_amount: int,
    target_amount: int,
    deadline: str,
    bull_analysis: str,
    bear_analysis: str,
    risk_analysis: str,
) -> str:
    logger.info("モデレーター統合分析開始")
    user = MODERATOR_USER_TEMPLATE.format(
        current_amount=current_amount,
        target_amount=target_amount,
        deadline=deadline,
        bull_analysis=bull_analysis,
        bear_analysis=bear_analysis,
        risk_analysis=risk_analysis,
    )
    return _call_agent(client, MODERATOR_SYSTEM, user)


# ── 自信度パース ────────────────────────────────────────────────────────────

def _extract_confidence(final_proposal: str) -> int:
    """モデレーター出力から自信度(%)を抽出する"""
    m = re.search(r"自信度.*?(\d{1,3})\s*%", final_proposal)
    if m:
        return int(m.group(1))
    return 50  # デフォルト


# ── データサマリー生成 ──────────────────────────────────────────────────────

def _build_data_summary(raw_data: dict) -> str:
    """raw_dataから簡潔なテキストサマリーを生成する（リスク管理官向け）"""
    lines = []
    for ticker, stock_data in raw_data.get("stocks", {}).items():
        tech = stock_data.get("technical", {})
        lines.append(f"[{ticker}]")
        lines.append(f"  現在株価: {tech.get('current_price')}円")
        lines.append(f"  前日比: {tech.get('price_change_pct')}%")
        lines.append(f"  RSI14: {tech.get('rsi14')}")
        macd = tech.get("macd", {})
        lines.append(f"  MACD: {macd.get('macd')} / Signal: {macd.get('signal')}")
        lines.append(f"  出来高MA比: {tech.get('volume_ma20_ratio')}")
    return "\n".join(lines) if lines else "データなし"


# ── メインエントリポイント ──────────────────────────────────────────────────

def generate_proposal(settings: dict) -> dict:
    """
    マルチエージェント議論を実行し、最終提案を返す。

    Args:
        settings: {
            "capital": 1000000,
            "current_amount": 1050000,
            "target_amount": 1500000,
            "deadline": "2025-12-31",
            "stocks": ["7203.T", "6758.T"],
            "stock_names": {"7203.T": "トヨタ自動車", "6758.T": "ソニーグループ"},
        }

    Returns:
        {
            "raw_data": {...},
            "web_search_result": "...",
            "bull_analysis": "...",
            "bear_analysis": "...",
            "risk_analysis": "...",
            "final_proposal": "...",
            "confidence": 72,
        }
    """
    client = _get_client()
    tickers: list[str] = settings["stocks"]
    stock_names: dict = settings.get("stock_names", {})
    today_str = date.today().isoformat()

    # ── Step 1: データ収集 ──────────────────────────────────────────────────
    logger.info("Step1: データ収集開始")
    raw_data = collect_all_data(tickers)

    # web_search クエリ組み立て
    queries = []
    for ticker in tickers:
        company = stock_names.get(ticker, ticker)
        queries.extend(build_search_queries(company, ticker.replace(".T", "")))
    # 重複排除・最大15クエリ
    queries = list(dict.fromkeys(queries))[:15]

    logger.info(f"Step1: web_search ({len(queries)}クエリ)")
    web_search_result = _web_search_supplement(client, queries)
    raw_data["web_search"] = web_search_result

    # エージェントに渡すデータテキスト
    data_text = json.dumps(raw_data, ensure_ascii=False, indent=2)
    context_text = (
        f"対象銘柄: {', '.join(f'{t}({stock_names.get(t, t)})' for t in tickers)}\n"
        f"現在資産: {settings['current_amount']:,}円\n"
        f"目標金額: {settings['target_amount']:,}円\n"
        f"期日: {settings['deadline']}\n"
        f"今日: {today_str}"
    )
    data_summary = _build_data_summary(raw_data)

    # ── Step 2: 強気アナリスト ──────────────────────────────────────────────
    bull_analysis = run_bull_agent(client, context_text, data_text)

    # ── Step 3: 弱気アナリスト ──────────────────────────────────────────────
    bear_analysis = run_bear_agent(client, context_text, data_text)

    # ── Step 4: リスク管理官 ────────────────────────────────────────────────
    risk_analysis = run_risk_agent(
        client,
        capital=settings["capital"],
        current_amount=settings["current_amount"],
        target_amount=settings["target_amount"],
        deadline=settings["deadline"],
        today=today_str,
        data_summary=data_summary,
    )

    # ── Step 5: モデレーター統合 ────────────────────────────────────────────
    final_proposal = run_moderator_agent(
        client,
        current_amount=settings["current_amount"],
        target_amount=settings["target_amount"],
        deadline=settings["deadline"],
        bull_analysis=bull_analysis,
        bear_analysis=bear_analysis,
        risk_analysis=risk_analysis,
    )

    confidence = _extract_confidence(final_proposal)

    logger.info(f"提案生成完了 (自信度: {confidence}%)")

    return {
        "raw_data": raw_data,
        "web_search_result": web_search_result,
        "bull_analysis": bull_analysis,
        "bear_analysis": bear_analysis,
        "risk_analysis": risk_analysis,
        "final_proposal": final_proposal,
        "confidence": confidence,
    }
