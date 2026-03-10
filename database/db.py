"""
データベース操作モジュール
- ローカル: SQLite（./kabu_advisor.db）
- Render: PostgreSQL（DATABASE_URL 環境変数で切り替え）
"""

import logging
import os
from datetime import date, datetime
from typing import Optional

from sqlalchemy import (
    Column, Integer, String, Text, Date, DateTime, Float, ForeignKey, create_engine
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session

logger = logging.getLogger(__name__)

_raw_url = os.getenv("DATABASE_URL", "sqlite:///./kabu_advisor.db")
# Render は "postgres://" で提供するが SQLAlchemy は "postgresql://" が必要
DATABASE_URL = _raw_url.replace("postgres://", "postgresql://", 1)

if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
else:
    engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ── モデル定義 ──────────────────────────────────────────────────────────────

class Settings(Base):
    __tablename__ = "settings"

    id = Column(Integer, primary_key=True)
    capital = Column(Integer)           # 元手
    current_cash = Column(Integer)      # 現在の現金（株を除く手元資金）
    target_amount = Column(Integer)     # 目標金額
    deadline = Column(Date)             # 期日
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Portfolio(Base):
    """現在の保有銘柄"""
    __tablename__ = "portfolio"

    id = Column(Integer, primary_key=True)
    ticker = Column(String(20), nullable=False)   # "7203.T"
    company_name = Column(String(100))
    shares = Column(Integer, nullable=False)       # 保有株数
    avg_price = Column(Float, nullable=False)      # 平均取得単価
    bought_date = Column(Date)
    created_at = Column(DateTime, default=datetime.utcnow)


class Proposal(Base):
    __tablename__ = "proposals"

    id = Column(Integer, primary_key=True)
    date = Column(Date, index=True)
    raw_data = Column(Text)             # 収集した全データ（JSON）
    screening_result = Column(Text)     # スクリーニング結果
    bull_analysis = Column(Text)
    bear_analysis = Column(Text)
    risk_analysis = Column(Text)
    final_proposal = Column(Text)
    confidence = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)


class TradeResult(Base):
    """実際に行ったトレードの記録"""
    __tablename__ = "trade_results"

    id = Column(Integer, primary_key=True)
    date = Column(Date)
    proposal_id = Column(Integer, ForeignKey("proposals.id"))
    ticker = Column(String(20))
    company_name = Column(String(100))
    action = Column(String(10))         # "buy" / "sell" / "hold"
    shares = Column(Integer)
    price = Column(Float)
    pnl = Column(Integer)               # 損益（売りのみ）
    memo = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)


def init_db():
    Base.metadata.create_all(bind=engine)
    logger.info("DB初期化完了")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ── Settings CRUD ──────────────────────────────────────────────────────────

def upsert_settings(db: Session, data: dict) -> Settings:
    existing = db.query(Settings).order_by(Settings.id.desc()).first()
    if existing:
        for k, v in data.items():
            setattr(existing, k, v)
        existing.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(existing)
        return existing
    row = Settings(**data)
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def get_settings(db: Session) -> Optional[Settings]:
    return db.query(Settings).order_by(Settings.id.desc()).first()


# ── Portfolio CRUD ─────────────────────────────────────────────────────────

def get_portfolio(db: Session) -> list[Portfolio]:
    return db.query(Portfolio).all()


def add_position(db: Session, ticker: str, company_name: str, shares: int, avg_price: float) -> Portfolio:
    # 既存ポジションがあれば平均単価を更新
    existing = db.query(Portfolio).filter(Portfolio.ticker == ticker).first()
    if existing:
        total_shares = existing.shares + shares
        existing.avg_price = (existing.avg_price * existing.shares + avg_price * shares) / total_shares
        existing.shares = total_shares
        db.commit()
        db.refresh(existing)
        return existing
    row = Portfolio(
        ticker=ticker,
        company_name=company_name,
        shares=shares,
        avg_price=avg_price,
        bought_date=date.today(),
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def reduce_position(db: Session, ticker: str, shares: int) -> Optional[Portfolio]:
    existing = db.query(Portfolio).filter(Portfolio.ticker == ticker).first()
    if not existing:
        return None
    existing.shares -= shares
    if existing.shares <= 0:
        db.delete(existing)
    db.commit()
    return existing


def clear_portfolio(db: Session):
    db.query(Portfolio).delete()
    db.commit()


# ── Proposal CRUD ──────────────────────────────────────────────────────────

def save_proposal(db: Session, data: dict) -> Proposal:
    row = Proposal(**data)
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def get_proposal_by_date(db: Session, target_date: date) -> Optional[Proposal]:
    return db.query(Proposal).filter(Proposal.date == target_date).first()


def get_latest_proposal(db: Session) -> Optional[Proposal]:
    return db.query(Proposal).order_by(Proposal.id.desc()).first()


def list_proposals(db: Session, limit: int = 30) -> list[Proposal]:
    return db.query(Proposal).order_by(Proposal.date.desc()).limit(limit).all()


# ── TradeResult CRUD ───────────────────────────────────────────────────────

def save_trade_result(db: Session, data: dict) -> TradeResult:
    row = TradeResult(**data)
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def list_trade_results(db: Session, limit: int = 60) -> list[TradeResult]:
    return db.query(TradeResult).order_by(TradeResult.date.desc()).limit(limit).all()


def get_total_pnl(db: Session) -> int:
    results = db.query(TradeResult).filter(TradeResult.pnl != None).all()
    return sum(r.pnl for r in results if r.pnl)
