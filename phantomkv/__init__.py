"""Phantom: deterministic simulation testing for LLM-serving KV caches."""

from .engine import PLANTS, ToyEngine, Violation
from .provenance import Tag
from .sim import Config, Result, Simulator, run_seed

__version__ = "0.5.0"
__all__ = ["PLANTS", "Config", "Result", "Simulator", "Tag", "ToyEngine",
           "Violation", "run_seed"]
