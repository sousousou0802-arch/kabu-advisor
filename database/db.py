"""
SQLite データベース操作モジュール
"""

import json
import logging
from datetime import date, datetime
from typing import Optional

from sqlalchemy import (
    Column, Integer, String, Text, Date, DateTime, ForeignKey, create_engine
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session

logger = logging.getLogger(__name__)

DATABASE_URL = "sqlite:///./kabu_advisor.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ── モデル定義 ──────────────────────────────────────────────────────────────

class Settings(Base):
    __tablename__ = "settings"

    id = Column(Integer, primary_key=True, index=True)
    capital = Column(Integer)           # 元手
    current_amount = Column(Integer)    # 現在資産
    target_amount = Column(Integer)     # 目標金額
    deadline = Column(Date)             # 期日
    stocks = Column(Text)               # 対象銘柄（JSON配列）
    created_at = Column(DateTime, default=datetime.utcnow)


class Proposal(Base):
    __tablename__ = "proposals"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date, index=True)
    raw_data = Column(Text)             # 収集した全データ（JSON）
    bull_analysis = Column(Text)        # 強気アナリスト分析
    bear_analysis = Column(Text)        # 弱気アナリスト分析
    risk_analysis = Column(Text)        # リスク管理官分析
    final_proposal = Column(Text)       # モデレーター最終提案
    confidence = Column(Integer)        # 自信度（%）
    created_at = Column(DateTime, default=datetime.utcnow)


class Result(Base):
    __tablename__ = "results"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date)
    proposal_id = Column(Integer, ForeignKey("proposals.id"))
    pnl = Column(Integer)               # 損益（円）
    memo = Column(Text)
    amount_after = Column(Integer)      # 取引後資産
    created_at = Column(DateTime, default=datetime.utcnow)


def init_db():
    """テーブルを作成する（初回起動時）"""
    Base.metadata.create_all(bind=engine)
    logger.info("DB初期化完了")


# ── CRUD ────────────────────────────────────────────────────────────────────

def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Settings

def upsert_settings(db: Session, data: dict) -> Settings:
    existing = db.query(Settings).order_by(Settings.id.desc()).first()
    if existing:
        for k, v in data.items():
            setattr(existing, k, v)
        db.commit()
        db.refresh(existing)
        return existing
    else:
        row = Settings(**data)
        db.add(row)
        db.commit()
        db.refresh(row)
        return row


def get_settings(db: Session) -> Optional[Settings]:
    return db.query(Settings).order_by(Settings.id.desc()).first()


# Proposal

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


# Result

def save_result(db: Session, data: dict) -> Result:
    row = Result(**data)
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def list_results(db: Session, limit: int = 30) -> list[Result]:
    return db.query(Result).order_by(Result.date.desc()).limit(limit).all()
