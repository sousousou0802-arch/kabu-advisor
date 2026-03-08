"""
FastAPI エントリポイント
"""

import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import date, datetime
from typing import Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy.orm import Session

load_dotenv()

from advisor.agents import generate_proposal
from database.db import (
    Result,
    Settings,
    get_db,
    get_latest_proposal,
    get_proposal_by_date,
    get_settings,
    init_db,
    list_proposals,
    list_results,
    save_proposal,
    save_result,
    upsert_settings,
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


# ── HTML ─────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    with open(html_path, encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


# ── Pydantic スキーマ ─────────────────────────────────────────────────────────

class SetupRequest(BaseModel):
    capital: int
    current_amount: int
    target_amount: int
    deadline: str          # "YYYY-MM-DD"
    stocks: list[str]      # ["7203.T", "6758.T"]
    stock_names: dict[str, str] = {}  # {"7203.T": "トヨタ自動車"}


class ResultRequest(BaseModel):
    proposal_id: int
    pnl: int
    memo: str = ""
    amount_after: int


# ── API ───────────────────────────────────────────────────────────────────────

@app.get("/api/status")
def api_status(db: Session = Depends(get_db)):
    settings = get_settings(db)
    if not settings:
        return {"configured": False}

    latest = get_latest_proposal(db)
    today_str = date.today().isoformat()
    has_today = latest and str(latest.date) == today_str

    stocks = json.loads(settings.stocks) if settings.stocks else []

    # 残り日数
    deadline_date = settings.deadline
    remaining_days = (deadline_date - date.today()).days if deadline_date else None

    # 進捗率
    progress_pct = None
    if settings.capital and settings.target_amount and settings.capital < settings.target_amount:
        gained = (settings.current_amount or settings.capital) - settings.capital
        needed = settings.target_amount - settings.capital
        progress_pct = round(gained / needed * 100, 1)

    return {
        "configured": True,
        "capital": settings.capital,
        "current_amount": settings.current_amount,
        "target_amount": settings.target_amount,
        "deadline": str(settings.deadline),
        "stocks": stocks,
        "remaining_days": remaining_days,
        "progress_pct": progress_pct,
        "has_today_proposal": bool(has_today),
        "latest_proposal_date": str(latest.date) if latest else None,
    }


@app.post("/api/setup")
def api_setup(req: SetupRequest, db: Session = Depends(get_db)):
    deadline = date.fromisoformat(req.deadline)
    row = upsert_settings(db, {
        "capital": req.capital,
        "current_amount": req.current_amount,
        "target_amount": req.target_amount,
        "deadline": deadline,
        "stocks": json.dumps(req.stocks, ensure_ascii=False),
    })
    return {"ok": True, "id": row.id}


@app.post("/api/generate")
def api_generate(db: Session = Depends(get_db)):
    settings = get_settings(db)
    if not settings:
        raise HTTPException(status_code=400, detail="設定が未完了です。/api/setup を先に呼んでください。")

    stocks = json.loads(settings.stocks) if settings.stocks else []
    if not stocks:
        raise HTTPException(status_code=400, detail="対象銘柄が設定されていません。")

    # 今日すでに生成済みの場合はそちらを返す
    today = date.today()
    existing = get_proposal_by_date(db, today)
    if existing:
        return {
            "ok": True,
            "cached": True,
            "proposal_id": existing.id,
            "final_proposal": existing.final_proposal,
            "confidence": existing.confidence,
        }

    settings_dict = {
        "capital": settings.capital,
        "current_amount": settings.current_amount,
        "target_amount": settings.target_amount,
        "deadline": str(settings.deadline),
        "stocks": stocks,
        "stock_names": {},
    }

    try:
        result = generate_proposal(settings_dict)
    except Exception as e:
        logger.error(f"提案生成エラー: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    # DB保存
    proposal = save_proposal(db, {
        "date": today,
        "raw_data": json.dumps(result["raw_data"], ensure_ascii=False),
        "bull_analysis": result["bull_analysis"],
        "bear_analysis": result["bear_analysis"],
        "risk_analysis": result["risk_analysis"],
        "final_proposal": result["final_proposal"],
        "confidence": result["confidence"],
    })

    return {
        "ok": True,
        "cached": False,
        "proposal_id": proposal.id,
        "final_proposal": proposal.final_proposal,
        "confidence": proposal.confidence,
    }


@app.get("/api/proposal/today")
def api_proposal_today(db: Session = Depends(get_db)):
    today = date.today()
    proposal = get_proposal_by_date(db, today)
    if not proposal:
        return {"found": False}
    return {
        "found": True,
        "id": proposal.id,
        "date": str(proposal.date),
        "final_proposal": proposal.final_proposal,
        "bull_analysis": proposal.bull_analysis,
        "bear_analysis": proposal.bear_analysis,
        "risk_analysis": proposal.risk_analysis,
        "confidence": proposal.confidence,
        "created_at": proposal.created_at.isoformat() if proposal.created_at else None,
    }


@app.post("/api/result")
def api_result(req: ResultRequest, db: Session = Depends(get_db)):
    settings = get_settings(db)
    row = save_result(db, {
        "date": date.today(),
        "proposal_id": req.proposal_id,
        "pnl": req.pnl,
        "memo": req.memo,
        "amount_after": req.amount_after,
    })
    # 現在資産を更新
    if settings:
        upsert_settings(db, {"current_amount": req.amount_after})

    return {"ok": True, "id": row.id}


@app.get("/api/history")
def api_history(db: Session = Depends(get_db)):
    proposals = list_proposals(db, limit=30)
    results = list_results(db, limit=30)

    results_by_proposal = {}
    for r in results:
        results_by_proposal[r.proposal_id] = r

    history = []
    for p in proposals:
        r = results_by_proposal.get(p.id)
        history.append({
            "date": str(p.date),
            "proposal_id": p.id,
            "final_proposal": p.final_proposal,
            "confidence": p.confidence,
            "pnl": r.pnl if r else None,
            "amount_after": r.amount_after if r else None,
            "memo": r.memo if r else None,
        })

    return {"history": history}
