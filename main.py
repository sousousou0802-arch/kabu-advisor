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
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
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

# 全 API レスポンスにキャッシュ無効化ヘッダーを付ける
@app.middleware("http")
async def no_cache_middleware(request: Request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response

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



class TradeResultRequest(BaseModel):
    proposal_id: Optional[int] = None
    trades: list[dict]  # [{ticker, company_name, action, shares, price, pnl, memo}]
    new_cash: int = None  # 廃止（自動計算）

class EditPositionRequest(BaseModel):
    company_name: Optional[str] = None
    shares: Optional[int] = None
    avg_price: Optional[float] = None


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
    today_date = date.today()
    remaining_days = (deadline_date - today_date).days if deadline_date else None

    total_pnl = get_total_pnl(db)

    # 今日の記録済みトレードを確認
    today_trades = [t for t in list_trade_results(db, limit=30)
                    if str(t.date) == today_str]

    # 目標進捗・軌道計算
    capital = settings.capital or 0
    target_amount = settings.target_amount or 0
    current_cash = settings.current_cash or 0
    portfolio_value = int(sum(p.avg_price * p.shares for p in portfolio))
    total_assets = current_cash + portfolio_value

    goal_needed = target_amount - capital
    goal_achieved = total_assets - capital
    goal_progress_pct = round(goal_achieved / goal_needed * 100, 1) if goal_needed > 0 else 0.0

    # 経過日数割合（settings.created_at を開始日として使用）
    if settings.created_at and deadline_date:
        start_date = settings.created_at.date()
        total_days = (deadline_date - start_date).days
        elapsed_days = (today_date - start_date).days
        days_elapsed_pct = round(elapsed_days / total_days * 100, 1) if total_days > 0 else 0.0
        # 現ペースでの期日到達時予測
        if elapsed_days > 0:
            daily_gain = goal_achieved / elapsed_days
            projected_final = capital + daily_gain * total_days
        else:
            projected_final = total_assets
    else:
        days_elapsed_pct = 0.0
        projected_final = total_assets

    gap = goal_progress_pct - days_elapsed_pct
    if gap >= 10:
        trajectory = "超過達成ペース"
    elif gap >= -10:
        trajectory = "順調"
    elif gap >= -30:
        trajectory = "遅延"
    else:
        trajectory = "大幅遅延"

    return {
        "configured": True,
        "capital": capital,
        "current_cash": current_cash,
        "target_amount": target_amount,
        "deadline": str(settings.deadline),
        "remaining_days": remaining_days,
        "portfolio": portfolio_list,
        "total_pnl": total_pnl,
        "has_today_proposal": bool(has_today),
        "latest_proposal_date": str(latest.date) if latest else None,
        "today_recorded": len(today_trades) > 0,
        "today_trade_count": len(today_trades),
        "total_assets": total_assets,
        "goal_progress_pct": goal_progress_pct,
        "days_elapsed_pct": days_elapsed_pct,
        "trajectory": trajectory,
        "projected_final": int(projected_final),
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
    # 軌道計算（エージェントに渡すため再計算）
    _cap = settings.capital or 0
    _cash = settings.current_cash or 0
    _target = settings.target_amount or 0
    _pv = int(sum(p["avg_price"] * p["shares"] for p in portfolio_list))
    _total = _cash + _pv
    _goal_needed = _target - _cap
    _goal_achieved = _total - _cap
    _goal_pct = round(_goal_achieved / _goal_needed * 100, 1) if _goal_needed > 0 else 0.0
    if settings.created_at and settings.deadline:
        _start = settings.created_at.date()
        _total_days = (settings.deadline - _start).days
        _elapsed = (date.today() - _start).days
        _days_pct = round(_elapsed / _total_days * 100, 1) if _total_days > 0 else 0.0
    else:
        _days_pct = 0.0
    _gap = _goal_pct - _days_pct
    _traj = "超過達成ペース" if _gap >= 10 else "順調" if _gap >= -10 else "遅延" if _gap >= -30 else "大幅遅延"

    settings_dict = {
        "capital": settings.capital,
        "current_cash": settings.current_cash,
        "target_amount": settings.target_amount,
        "deadline": str(settings.deadline),
        "goal_progress_pct": _goal_pct,
        "days_elapsed_pct": _days_pct,
        "trajectory": _traj,
    }
    today = date.today()
    past_proposals = [p for p in list_proposals(db, limit=6) if p.date != today][:5]
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
    from database.db import Portfolio

    # 売り記録の現金計算のため、ポートフォリオ更新前に実際の保有株数を記録しておく
    pre_sell_shares: dict[str, int] = {}
    for trade in req.trades:
        if trade.get("action") == "sell":
            t = (trade.get("ticker") or "").upper()
            if t and t not in pre_sell_shares:
                pos = db.query(Portfolio).filter(Portfolio.ticker == t).first()
                pre_sell_shares[t] = pos.shares if pos else 0

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

    # 現金残高を取引内容から自動計算（全オブジェクトを明示的に期限切れにしてDB最新値を取得）
    db.expire_all()
    settings = get_settings(db)
    current_cash = (settings.current_cash or 0) if settings else 0
    for trade in req.trades:
        action = trade.get("action")
        ticker = (trade.get("ticker") or "").upper()
        shares = trade.get("shares", 0) or 0
        price = trade.get("price", 0) or 0
        if action == "buy":
            current_cash -= int(shares * price)
        elif action == "sell":
            # 保有記録がある場合のみ上限キャップ。ない場合はそのまま加算
            held = pre_sell_shares.get(ticker)
            actual = min(shares, held) if held else shares
            current_cash += int(actual * price)
    upsert_settings(db, {"current_cash": current_cash})

    return {"ok": True}


@app.patch("/api/portfolio/{ticker}")
def api_portfolio_edit(ticker: str, req: EditPositionRequest, db: Session = Depends(get_db)):
    from database.db import Portfolio
    pos = db.query(Portfolio).filter(Portfolio.ticker == ticker.upper()).first()
    if not pos:
        raise HTTPException(status_code=404, detail="銘柄が見つかりません")

    # 修正前のコスト
    old_cost = int(pos.shares * pos.avg_price)
    logger.info(f"[portfolio_edit] {ticker}: 変更前 shares={pos.shares} avg_price={pos.avg_price} old_cost={old_cost}")

    deleted = False
    if req.company_name is not None:
        pos.company_name = req.company_name
    if req.shares is not None:
        if req.shares <= 0:
            db.delete(pos)
            deleted = True
        else:
            pos.shares = req.shares
    if not deleted and req.avg_price is not None:
        pos.avg_price = req.avg_price

    # 修正後コスト（コミット前にメモリ上の値から計算）
    new_cost = 0 if deleted else int(pos.shares * pos.avg_price)
    cash_delta = old_cost - new_cost
    logger.info(f"[portfolio_edit] {ticker}: 変更後 shares={req.shares} avg_price={req.avg_price} new_cost={new_cost} cash_delta={cash_delta}")

    # ① ポートフォリオ変更を先にコミット
    db.commit()
    logger.info(f"[portfolio_edit] {ticker}: ポートフォリオ commit 完了")

    # ② 現金残高を更新（別トランザクション）
    if cash_delta != 0:
        settings = get_settings(db)
        before_cash = settings.current_cash if settings else None
        if settings:
            new_cash = (settings.current_cash or 0) + cash_delta
            upsert_settings(db, {"current_cash": new_cash})
            logger.info(f"[portfolio_edit] 現金更新: {before_cash} → {new_cash}")
    else:
        logger.info(f"[portfolio_edit] cash_delta=0 のため現金変更なし")

    return {"ok": True}


@app.get("/api/prices")
def api_prices(db: Session = Depends(get_db)):
    """保有銘柄の現在株価をyfinanceから取得する"""
    import yfinance as yf
    portfolio = get_portfolio(db)
    if not portfolio:
        return {"prices": {}}
    prices = {}
    for pos in portfolio:
        try:
            fi = yf.Ticker(pos.ticker).fast_info
            p = getattr(fi, "last_price", None) or getattr(fi, "previous_close", None)
            prices[pos.ticker] = round(float(p)) if p else None
        except Exception as e:
            logger.warning(f"価格取得失敗 {pos.ticker}: {e}")
            prices[pos.ticker] = None
    return {"prices": prices}


@app.get("/api/history")
def api_history(db: Session = Depends(get_db)):
    proposals = list_proposals(db, limit=30)
    trades = list_trade_results(db, limit=90)

    trades_by_proposal: dict[int, list] = {}
    orphan_trades: list = []  # proposal_id=None（手動記録）
    for t in trades:
        if t.proposal_id is None:
            orphan_trades.append(t)
        else:
            trades_by_proposal.setdefault(t.proposal_id, []).append(t)

    history = []
    seen_dates = set()
    for p in proposals:
        t_list = trades_by_proposal.get(p.id, [])
        # 同日の手動記録も合算して表示
        p_date_str = str(p.date)
        if p_date_str not in seen_dates:
            t_list = t_list + [t for t in orphan_trades if str(t.date) == p_date_str]
            seen_dates.add(p_date_str)
        pnl_values = [t.pnl for t in t_list if t.pnl is not None]
        day_pnl = sum(pnl_values) if pnl_values else None
        history.append({
            "date": p_date_str,
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
