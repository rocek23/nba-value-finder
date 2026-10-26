"""Backtesting rigoroso: split temporali, walk-forward, baseline, quote storiche pre-partita, archivio verificabile."""

from .runner import BacktestConfig, BacktestResult, run_backtest  # noqa: F401
from .splits import TemporalSplit, make_split  # noqa: F401
