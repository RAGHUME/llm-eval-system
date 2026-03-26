"""
Database Module — SQLAlchemy ORM + CRUD Operations
====================================================

WHAT: Defines the database schema and provides all create/read operations
      for storing evaluation runs, prompt results, and scores.

WHY:  We need persistent storage to track evaluation history, compare
      results over time, and enable the optimization pipeline to reference
      past performance data.

HOW:  Uses SQLAlchemy ORM with SQLite backend. Three tables:
      - EvaluationRun: stores query + metadata per evaluation session
      - PromptResult: stores each prompt's text, strategy, and LLM response
      - Score: stores all 7 metric scores + total + error flags

OUTPUT: Functions return dictionaries or lists of dictionaries for easy
        JSON serialization in the API layer.
"""

import json
import os
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import Session, declarative_base, relationship, sessionmaker

# ---------------------------------------------------------------------------
# Database setup — SQLite file stored in data/ directory
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "eval_results.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},  # Required for SQLite + FastAPI
    echo=False,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ===========================================================================
# SQLAlchemy Models
# ===========================================================================


class EvaluationRun(Base):
    """
    Represents a single evaluation session.
    One run can evaluate multiple prompts against the same query.
    """

    __tablename__ = "evaluation_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    query = Column(Text, nullable=False)
    reference_answer = Column(Text, nullable=True)
    model_used = Column(String(100), nullable=False, default="phi3:mini")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationship: one run → many prompt results
    results = relationship(
        "PromptResult", back_populates="run", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<EvaluationRun(id={self.id}, query='{self.query[:50]}...')>"


class PromptResult(Base):
    """
    Stores one prompt's text, strategy, version, LLM response, and rank.
    Each PromptResult belongs to exactly one EvaluationRun.
    """

    __tablename__ = "prompt_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(Integer, ForeignKey("evaluation_runs.id"), nullable=False)
    prompt_text = Column(Text, nullable=False)
    prompt_strategy = Column(String(50), nullable=True)  # zero_shot, few_shot, etc.
    prompt_version = Column(String(20), nullable=True, default="v1")
    llm_response = Column(Text, nullable=True)
    rank = Column(Integer, nullable=True)

    # Relationships
    run = relationship("EvaluationRun", back_populates="results")
    scores = relationship(
        "Score", back_populates="result", cascade="all, delete-orphan", uselist=False
    )

    def __repr__(self) -> str:
        return f"<PromptResult(id={self.id}, strategy='{self.prompt_strategy}', rank={self.rank})>"


class Score(Base):
    """
    Stores all 7 evaluation metric scores, total score, and error flags.
    Each Score belongs to exactly one PromptResult (one-to-one).
    """

    __tablename__ = "scores"

    id = Column(Integer, primary_key=True, autoincrement=True)
    result_id = Column(Integer, ForeignKey("prompt_results.id"), nullable=False)

    # Layer 1 — Basic metrics
    bleu = Column(Float, nullable=True, default=0.0)
    rouge = Column(Float, nullable=True, default=0.0)
    relevance = Column(Float, nullable=True, default=0.0)

    # Layer 2 — Intelligent analysis
    entity_score = Column(Float, nullable=True, default=0.0)
    structure_score = Column(Float, nullable=True, default=0.0)
    consistency_score = Column(Float, nullable=True, default=0.0)
    llm_judge_score = Column(Float, nullable=True, default=0.0)

    # Aggregated
    total_score = Column(Float, nullable=True, default=0.0)

    # Error detection
    hallucination_flag = Column(Boolean, nullable=True, default=False)
    error_flags = Column(Text, nullable=True, default="[]")  # JSON string

    # Relationship
    result = relationship("PromptResult", back_populates="scores")

    def __repr__(self) -> str:
        return f"<Score(id={self.id}, total={self.total_score:.2f})>"


# ===========================================================================
# Database Initialization
# ===========================================================================


def create_tables() -> None:
    """Create all database tables if they don't exist."""
    Base.metadata.create_all(bind=engine)


def get_db() -> Session:
    """
    Get a database session. Use as context manager:
        db = get_db()
        try:
            # ... do work ...
        finally:
            db.close()
    """
    return SessionLocal()


# ===========================================================================
# CRUD Operations
# ===========================================================================


def save_run(
    query: str,
    reference_answer: Optional[str] = None,
    model_used: str = "phi3:mini",
) -> int:
    """
    Create a new evaluation run and return its ID.

    Args:
        query: The user's question/topic being evaluated
        reference_answer: Optional gold-standard answer for comparison
        model_used: Which Ollama model was used

    Returns:
        The auto-generated run ID (integer)
    """
    db = get_db()
    try:
        run = EvaluationRun(
            query=query,
            reference_answer=reference_answer,
            model_used=model_used,
        )
        db.add(run)
        db.commit()
        db.refresh(run)
        return run.id
    finally:
        db.close()


def save_result(
    run_id: int,
    prompt_text: str,
    strategy: Optional[str] = None,
    version: str = "v1",
    response: Optional[str] = None,
) -> int:
    """
    Save a single prompt result within an evaluation run.

    Args:
        run_id: Foreign key to EvaluationRun
        prompt_text: The actual prompt sent to the LLM
        strategy: Prompt strategy type (zero_shot, few_shot, etc.)
        version: Prompt version label (v1, v2, etc.)
        response: The LLM's response text

    Returns:
        The auto-generated result ID (integer)
    """
    db = get_db()
    try:
        result = PromptResult(
            run_id=run_id,
            prompt_text=prompt_text,
            prompt_strategy=strategy,
            prompt_version=version,
            llm_response=response,
        )
        db.add(result)
        db.commit()
        db.refresh(result)
        return result.id
    finally:
        db.close()


def save_score(
    result_id: int,
    bleu: float = 0.0,
    rouge: float = 0.0,
    relevance: float = 0.0,
    entity_score: float = 0.0,
    structure_score: float = 0.0,
    consistency_score: float = 0.0,
    llm_judge_score: float = 0.0,
    total_score: float = 0.0,
    hallucination_flag: bool = False,
    error_flags: Optional[list] = None,
) -> int:
    """
    Save all metric scores for a prompt result.

    Args:
        result_id: Foreign key to PromptResult
        bleu..llm_judge_score: Individual metric scores (0.0-1.0)
        total_score: Weighted aggregate score
        hallucination_flag: Whether hallucination was detected
        error_flags: List of error string descriptions

    Returns:
        The auto-generated score ID (integer)
    """
    db = get_db()
    try:
        score = Score(
            result_id=result_id,
            bleu=bleu,
            rouge=rouge,
            relevance=relevance,
            entity_score=entity_score,
            structure_score=structure_score,
            consistency_score=consistency_score,
            llm_judge_score=llm_judge_score,
            total_score=total_score,
            hallucination_flag=hallucination_flag,
            error_flags=json.dumps(error_flags or []),
        )
        db.add(score)
        db.commit()
        db.refresh(score)
        return score.id
    finally:
        db.close()


def update_result_rank(result_id: int, rank: int) -> None:
    """Update the rank field for a prompt result."""
    db = get_db()
    try:
        result = db.query(PromptResult).filter(PromptResult.id == result_id).first()
        if result:
            result.rank = rank
            db.commit()
    finally:
        db.close()


def get_history() -> list[dict]:
    """
    Get all past evaluation runs with summary info.

    Returns:
        List of dicts with: id, query, model_used, created_at,
        num_prompts, best_score
    """
    db = get_db()
    try:
        runs = (
            db.query(EvaluationRun)
            .order_by(EvaluationRun.created_at.desc())
            .all()
        )
        history = []
        for run in runs:
            # Find best score among all results in this run
            best_score = 0.0
            num_prompts = len(run.results)
            for result in run.results:
                if result.scores and result.scores.total_score:
                    best_score = max(best_score, result.scores.total_score)

            history.append({
                "id": run.id,
                "query": run.query,
                "model_used": run.model_used,
                "created_at": run.created_at.strftime("%Y-%m-%d %H:%M") if run.created_at else "",
                "num_prompts": num_prompts,
                "best_score": round(best_score, 3),
            })
        return history
    finally:
        db.close()


def get_run_detail(run_id: int) -> Optional[dict]:
    """
    Get full details of a specific evaluation run, including all
    prompt results and their scores.

    Args:
        run_id: The evaluation run ID

    Returns:
        Dict with run metadata + list of results (each with scores),
        or None if run not found
    """
    db = get_db()
    try:
        run = db.query(EvaluationRun).filter(EvaluationRun.id == run_id).first()
        if not run:
            return None

        results = []
        for result in sorted(run.results, key=lambda r: r.rank or 999):
            score_data = {}
            error_list = []
            if result.scores:
                score_data = {
                    "bleu": round(result.scores.bleu or 0, 3),
                    "rouge": round(result.scores.rouge or 0, 3),
                    "relevance": round(result.scores.relevance or 0, 3),
                    "entity_score": round(result.scores.entity_score or 0, 3),
                    "structure_score": round(result.scores.structure_score or 0, 3),
                    "consistency_score": round(result.scores.consistency_score or 0, 3),
                    "llm_judge_score": round(result.scores.llm_judge_score or 0, 3),
                    "total_score": round(result.scores.total_score or 0, 3),
                    "hallucination_flag": result.scores.hallucination_flag or False,
                }
                try:
                    error_list = json.loads(result.scores.error_flags or "[]")
                except json.JSONDecodeError:
                    error_list = []

            results.append({
                "id": result.id,
                "prompt_text": result.prompt_text,
                "prompt_strategy": result.prompt_strategy,
                "prompt_version": result.prompt_version,
                "llm_response": result.llm_response,
                "rank": result.rank,
                "scores": score_data,
                "error_flags": error_list,
            })

        return {
            "id": run.id,
            "query": run.query,
            "reference_answer": run.reference_answer,
            "model_used": run.model_used,
            "created_at": run.created_at.strftime("%Y-%m-%d %H:%M") if run.created_at else "",
            "results": results,
            "num_prompts": len(results),
            "best_score": max((r["scores"].get("total_score", 0) for r in results), default=0),
        }
    finally:
        db.close()


# Initialize tables on module import
create_tables()
