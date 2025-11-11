import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable


@dataclass
class PhaseResult:
    tokens: int
    seconds: float

    @property
    def tok_per_sec(self) -> float:
        if self.seconds <= 0:
            return float("inf")
        return self.tokens / self.seconds


class TokenTimer:
    """Utility to keep timing/throughput reporting consistent across scripts."""

    def __init__(self) -> None:
        self._phases: dict[str, PhaseResult] = {}

    @contextmanager
    def measure(self, name: str, token_count: Callable[[], int]):
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            tokens = token_count()
            self._phases[name] = PhaseResult(tokens=tokens, seconds=elapsed)

    def report(self) -> None:
        for name in ("prefill", "decode"):
            if name not in self._phases:
                continue
            phase = self._phases[name]
            print(
                f"[{name:7}] {phase.tokens} toks in {phase.seconds:.3f}s  → {phase.tok_per_sec:.1f} tok/s"
            )

    def get(self, name: str) -> PhaseResult | None:
        return self._phases.get(name)

    def phases(self) -> dict[str, PhaseResult]:
        return dict(self._phases)
