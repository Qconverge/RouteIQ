"""
Phase 7 — Database Models (SQLite via SQLModel)
================================================
No paid DB hosting — SQLite file at data/routeiq.db.
Uses SQLModel (wraps SQLAlchemy + Pydantic) for zero-boilerplate ORM.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from sqlmodel import Field, Session, SQLModel, create_engine, select

# ---------------------------------------------------------------------------
# Engine — SQLite, no hosting required
# ---------------------------------------------------------------------------
_DB_DIR = Path(__file__).parent.parent / "data"
_DB_DIR.mkdir(exist_ok=True)
DATABASE_URL = f"sqlite:///{_DB_DIR / 'routeiq.db'}"

engine = create_engine(
    DATABASE_URL,
    echo=False,
    connect_args={"check_same_thread": False},
)


def create_db() -> None:
    """Create all tables (idempotent)."""
    SQLModel.metadata.create_all(engine)


def get_session():
    """FastAPI dependency — yields a DB session."""
    with Session(engine) as session:
        yield session


# ---------------------------------------------------------------------------
# Scenario table
# ---------------------------------------------------------------------------

class ScenarioBase(SQLModel):
    name:           str   = Field(default="Unnamed Scenario")
    area_name:      str   = Field(default="Navrangpura, Ahmedabad")
    center_lat:     float = Field(default=23.0285)
    center_lon:     float = Field(default=72.5546)
    radius_m:       float = Field(default=1000.0)
    num_vehicles:   int   = Field(default=10)
    k_candidates:   int   = Field(default=5)
    # Lifecycle
    status:         str   = Field(default="created")   # created|optimising|complete|error
    current_stage:  str   = Field(default="")
    created_at:     datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at:     datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Scenario(ScenarioBase, table=True):
    id:              Optional[int] = Field(default=None, primary_key=True)
    od_pairs_json:   str = Field(default="[]")   # JSON [[o,d], ...]
    assignment_json: str = Field(default="[]")   # JSON [k, k, ...]
    metrics_json:    str = Field(default="{}")   # JSON metrics dict
    error_msg:       str = Field(default="")


class ScenarioCreate(ScenarioBase):
    pass


class ScenarioRead(ScenarioBase):
    id:              int
    od_pairs_json:   str
    assignment_json: str
    metrics_json:    str
    error_msg:       str


# ---------------------------------------------------------------------------
# Request / response bodies
# ---------------------------------------------------------------------------

class OptimiseRequest(SQLModel):
    seed:                   int = Field(default=0)
    n_vehicles_congestion:  int = Field(default=200)


class SimulateTrafficRequest(SQLModel):
    """
    Inject synthetic congestion — replaces live traffic API.
    Congestion is BPR-simulated, never live data.
    """
    scenario_type: str = Field(default="rush_hour")   # normal|moderate|heavy|blockage
    num_vehicles:  int = Field(default=100)
    seed:          int = Field(default=42)


class BenchmarkResult(SQLModel):
    algorithm:              str
    total_travel_time_s:    float
    mean_travel_time_s:     float
    congestion_cost:        float
    runtime_s:              float
    n_constraint_violations: int
    mean_path_length_m:     float
