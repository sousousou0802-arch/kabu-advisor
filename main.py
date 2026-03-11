"""
FastAPI エントリポイント
"""

import json
import logging
import os
import threading
from contextlib import asynccontextmanager
from datetime import date, datetime
from typing import Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

load_dotenv()

from advisor.agents import generate_proposal
from database.db import (
    get_db, init_db, SessionLocal,
    get_settings, upsert_settings,
    get_portfolio, add_position, reduce_position,
    save_proposal, get_proposal_by_date, get_latest_proposal, list_proposals,
    save_trade_result, list_trade_results, get_total_pnl,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="kabu-advisor", lifespan=lifespan)

# バックグラウンド生成ジョブの状態管理（プロセス内メモリ）
_job: dict = {"running": False, "error": None, "started_at": None}


# ── HTML ──────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    with open(html_path, encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


# ── Pydantic スキーマ ──────────────────────────────────────────────────────────

class SetupRequest(BaseModel):
    capital: int
    current_cash: int
    target_amount: int
    deadline: str  # "YYYY-MM-DD"


class AddPositionRequest(BaseModel):
    ticker: str
    company_name: str = ""
    shares: int
    avg_price: float


class TradeResultRequest(BaseModel):
    proposal_id: Optional[int] = None
    trades: list[dict]  # [{ticker, company_name, action, shares, price, pnl, memo}]
    new_cash: int = None  # 廃止（自動計算）


# ── API ───────────────────────────────────────────────────────────────────────

@app.get("/api/status")
def api_status(db: Session = Depends(get_db)):
    settings = get_settings(db)
    if not settings:
        return {"configured": False}

    portfolio = get_portfolio(db)
    portfolio_list = [
        {"ticker": p.ticker, "company_name": p.company_name,
         "shares": p.shares, "avg_price": p.avg_price}
        for p in portfolio
    ]

    latest = get_latest_proposal(db)
    today_str = date.today().isoformat()
    has_today = latest and str(latest.date) == today_str

    deadline_date = settings.deadline
    remaining_days = (deadline_date - date.today()).days if deadline_date else None

    total_pnl = get_total_pnl(db)

    return {
        "configured": True,
        "capital": settings.capital,
        "current_cash": settings.current_cash,
        "target_amount": settings.target_amount,
        "deadline": str(settings.deadline),
        "remaining_days": remaining_days,
        "portfolio": portfolio_list,
        "total_pnl": total_pnl,
        "has_today_proposal": bool(has_today),
        "latest_proposal_date": str(latest.date) if latest else None,
    }


@app.post("/api/setup")
def api_setup(req: SetupRequest, db: Session = Depends(get_db)):
    row = upsert_settings(db, {
        "capital": req.capital,
        "current_cash": req.current_cash,
        "target_amount": req.target_amount,
        "deadline": date.fromisoformat(req.deadline),
    })
    return {"ok": True, "id": row.id}


@app.post("/api/generate")
def api_generate(db: Session = Depends(get_db)):
    settings = get_settings(db)
    if not settings:
        raise HTTPException(status_code=400, detail="設定が未完了です。/api/setup を先に呼んでください。")

    today = date.today()

    # すでにバックグラウンドで生成中なら待機中を返す
    if _job["running"]:
        return {"ok": True, "status": "running"}

    # バックグラウンドスレッドで生成開始（Renderの30秒タイムアウトを回避）
    portfolio = get_portfolio(db)
    portfolio_list = [
        {"ticker": p.ticker, "company_name": p.company_name,
         "shares": p.shares, "avg_price": p.avg_price}
        for p in portfolio
    ]
    settings_dict = {
        "capital": settings.capital,
        "current_cash": settings.current_cash,
        "target_amount": settings.target_amount,
        "deadline": str(settings.deadline),
    }
    past_proposals = list_proposals(db, limit=5)
    past_trades = list_trade_results(db, limit=30)
    trades_by_proposal = {}
    for t in past_trades:
        trades_by_proposal.setdefault(t.proposal_id, []).append({
            "ticker": t.ticker, "action": t.action,
            "shares": t.shares, "price": t.price, "pnl": t.pnl,
        })
    history_list = [
        {
            "date": str(p.date),
            "final_proposal": p.final_proposal,
            "confidence": p.confidence,
            "trades": trades_by_proposal.get(p.id, []),
        }
        for p in past_proposals
    ]

    def _run():
        try:
            result = generate_proposal(settings_dict, portfolio_list, history_list)
            with SessionLocal() as bg_db:
                # 先に新しい提案を保存してから古いものを削除（生成失敗時にデータ消失しないよう順序を守る）
                save_proposal(bg_db, {
                    "date": today,
                    "raw_data": json.dumps(result["raw_data"], ensure_ascii=False),
                    "screening_result": result["screening_result"],
                    "bull_analysis": result["bull_analysis"],
                    "bear_analysis": result["bear_analysis"],
                    "risk_analysis": result["risk_analysis"],
                    "final_proposal": result["final_proposal"],
                    "confidence": result["confidence"],
                })
            logger.info("バックグラウンド提案生成完了")
        except Exception as e:
            logger.error(f"バックグラウンド提案生成エラー: {e}", exc_info=True)
            _job["error"] = str(e)
        finally:
            _job["running"] = False

    _job["running"] = True
    _job["error"] = None
    _job["started_at"] = datetime.now().isoformat()
    threading.Thread(target=_run, daemon=True).start()
    return {"ok": True, "status": "running"}


@app.get("/api/generate/status")
def api_generate_status(db: Session = Depends(get_db)):
    """バックグラウンド生成ジョブの状態を返す。フロントがポーリングで使用。"""
    # 生成中フラグを最優先でチェック（再生成中はDBに古い提案があっても running を返す）
    if _job["running"]:
        return {"status": "running", "started_at": _job["started_at"]}
    if _job["error"]:
        return {"status": "error", "detail": _job["error"]}
    # 最新提案があればdone（日付問わず）。自動再生成の誤発火を防ぐ
    latest = get_latest_proposal(db)
    if latest:
        return {"status": "done", "proposal_id": latest.id}
    return {"status": "idle"}


@app.get("/api/proposal/today")
def api_proposal_today(db: Session = Depends(get_db)):
    # 今日の提案を優先、なければ最新の提案を返す（UTC/JST日付ずれ対策）
    proposal = get_proposal_by_date(db, date.today()) or get_latest_proposal(db)
    if not proposal:
        return {"found": False}
    return {
        "found": True,
        "id": proposal.id,
        "date": str(proposal.date),
        "final_proposal": proposal.final_proposal,
        "screening_result": proposal.screening_result,
        "bull_analysis": proposal.bull_analysis,
        "bear_analysis": proposal.bear_analysis,
        "risk_analysis": proposal.risk_analysis,
        "confidence": proposal.confidence,
        "created_at": proposal.created_at.isoformat() if proposal.created_at else None,
    }


@app.post("/api/result")
def api_result(req: TradeResultRequest, db: Session = Depends(get_db)):
    # トレード結果を記録
    for trade in req.trades:
        save_trade_result(db, {
            "date": date.today(),
            "proposal_id": req.proposal_id,
            "ticker": trade.get("ticker"),
            "company_name": trade.get("company_name"),
            "action": trade.get("action"),
            "shares": trade.get("shares"),
            "price": trade.get("price"),
            "pnl": trade.get("pnl"),
            "memo": trade.get("memo", ""),
        })
        # ポートフォリオを更新
        action = trade.get("action")
        ticker = trade.get("ticker")
        shares = trade.get("shares", 0)
        price = trade.get("price", 0)
        if action == "buy" and ticker and shares and price:
            add_position(db, ticker, trade.get("company_name", ""), shares, price)
        elif action == "sell" and ticker and shares:
            reduce_position(db, ticker, shares)

    # 現金残高を取引内容から自動計算
    settings = get_settings(db)
    current_cash = settings.current_cash or 0
    for trade in req.trades:
        action = trade.get("action")
        shares = trade.get("shares", 0) or 0
        price = trade.get("price", 0) or 0
        pnl = trade.get("pnl", 0) or 0
        if action == "buy":
            current_cash -= int(shares * price)
        elif action == "sell":
            # 売却代金（shares×price）を加算、損益は約定価格に含まれるため別途加算しない
            current_cash += int(shares * price)
    upsert_settings(db, {"current_cash": current_cash})

    return {"ok": True}


@app.post("/api/portfolio/add")
def api_portfolio_add(req: AddPositionRequest, db: Session = Depends(get_db)):
    """保有銘柄を手動追加（取引記録なし）"""
    pos = add_position(db, req.ticker, req.company_name, req.shares, req.avg_price)
    return {"ok": True, "ticker": pos.ticker, "shares": pos.shares, "avg_price": pos.avg_price}


@app.delete("/api/portfolio/{ticker}")
def api_portfolio_delete(ticker: str, db: Session = Depends(get_db)):
    """保有銘柄を手動削除"""
    from database.db import Portfolio
    pos = db.query(Portfolio).filter(Portfolio.ticker == ticker.upper()).first()
    if not pos:
        raise HTTPException(status_code=404, detail="銘柄が見つかりません")
    db.delete(pos)
    db.commit()
    return {"ok": True}


@app.get("/api/history")
def api_history(db: Session = Depends(get_db)):
    proposals = list_proposals(db, limit=30)
    trades = list_trade_results(db, limit=90)

    trades_by_proposal: dict[int, list] = {}
    for t in trades:
        trades_by_proposal.setdefault(t.proposal_id, []).append(t)

    history = []
    for p in proposals:
        t_list = trades_by_proposal.get(p.id, [])
        day_pnl = sum(t.pnl for t in t_list if t.pnl)
        history.append({
            "date": str(p.date),
            "proposal_id": p.id,
            "final_proposal": p.final_proposal,
            "confidence": p.confidence,
            "trades": [
                {"ticker": t.ticker, "company_name": t.company_name,
                 "action": t.action, "shares": t.shares,
                 "price": t.price, "pnl": t.pnl}
                for t in t_list
            ],
            "day_pnl": day_pnl,
        })

    return {"history": history}
