"""
Day 3 - Statistical Analysis Toolkit
Squad Python Specialization - Data Science, AI & Analytics

RULE: implement every calculation yourself with plain Python.
NumPy may be imported ONLY in compare_with_numpy(), to verify your results.
Do not use the statistics module.
"""

import csv
import math
import re
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "agent_deposits.csv"
REPORT_PATH = BASE_DIR / "output" / "stats_report.txt"
COLUMN = "daily_deposits_kes"

QUARTILE_METHOD = "linear"   
NUMPY_DDOF = 1               
TOLERANCE = 1e-9             


PLACEHOLDERS = {"-", "--", "n/a", "na", "null", "none", "nan", "?", "x"}
_NUMBER_RE = re.compile(r"[+-]?\d+(\.\d+)?")



def check_value(raw: str | None) -> tuple[float | None, str | None]:
    """Return (value, None) if raw is trustworthy, else (None, reason).

    Nothing is guessed: "12,40O" is rejected, not "repaired" to 12400.
    """
    text = "" if raw is None else raw.strip()
    if text == "":
        return None, "missing value"
    if text.lower() in PLACEHOLDERS:
        return None, "placeholder, not a number"
    if not _NUMBER_RE.fullmatch(text):
        return None, "not a valid number, contains letters or stray characters"
    value = float(text)
    if value < 0:
        return None, "negative deposit is impossible"
    return value, None


def explain_rejection(raw: str) -> str:
    """The reason a raw value was rejected (used by the report)."""
    return check_value(raw)[1] or "valid"


def _scan(path: Path, column: str):
    """Read the CSV once. Returns (valid, rejected, rows_read).

    valid    : list of (value, (date, agent_id))
    rejected : list of (row_number, raw_value); row_number is the line number
               in the file, where the header is line 1.
    """
    valid, rejected, rows_read = [], [], 0
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or column not in reader.fieldnames:
            raise KeyError(f"column {column!r} not found in {path}")
        for row in reader:
            rows_read += 1
            raw = row.get(column)
            value, _reason = check_value(raw)
            if value is None:
                rejected.append((reader.line_num, raw or ""))
            else:
                valid.append((value, (row.get("date", ""), row.get("agent_id", ""))))
    return valid, rejected, rows_read


def load_column(path: Path, column: str) -> tuple[list[float], list[tuple[int, str]]]:
    """Return (valid_values, rejected) where rejected holds (row_number, raw_value) pairs."""
    valid, rejected, _ = _scan(Path(path), column)
    return [value for value, _label in valid], rejected



def _require(values) -> None:
    if len(values) == 0:
        raise ValueError("need at least one value")


def mean(values: list[float]) -> float:
    _require(values)
    return sum(values) / len(values)


def _median_sorted(s: list[float]) -> float:
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 == 1 else (s[mid - 1] + s[mid]) / 2


def median(values: list[float]) -> float:
    _require(values)
    return _median_sorted(sorted(values))


def mode(values: list[float]) -> list[float]:
    """Every value that occurs most often, in ascending order.

    If every value appears exactly once there is no meaningful mode, so the
    result is the empty list [].  (This also covers a single value.)
    """
    _require(values)
    counts: dict[float, int] = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    top = max(counts.values())
    if top == 1:
        return []
    return sorted(v for v, c in counts.items() if c == top)



def variance(values: list[float], sample: bool = True) -> float:
    """Sample variance (n - 1) by default; population variance (n) when sample is False."""
    _require(values)
    n = len(values)
    if sample and n < 2:
        raise ValueError("sample variance needs at least two values")
    m = mean(values)
    sum_sq = sum((v - m) ** 2 for v in values)
    return sum_sq / (n - 1 if sample else n)


def std_dev(values: list[float], sample: bool = True) -> float:
    return math.sqrt(variance(values, sample))



def _percentile_sorted(s: list[float], p: float) -> float:
    """Linear interpolation: position = (n - 1) * p (Hyndman & Fan type 7)."""
    pos = (len(s) - 1) * p
    lo = math.floor(pos)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def _quartiles_sorted(s: list[float]) -> tuple[float, float, float]:
    return _percentile_sorted(s, 0.25), _percentile_sorted(s, 0.5), _percentile_sorted(s, 0.75)


def quartiles(values: list[float]) -> tuple[float, float, float]:
    """Return (Q1, Q2, Q3) using linear interpolation (method name: "linear")."""
    _require(values)
    return _quartiles_sorted(sorted(values))


def _outliers_from_quartiles(values: list[float], q1: float, q3: float, k: float) -> dict:
    iqr = q3 - q1
    lower, upper = q1 - k * iqr, q3 + k * iqr
    return {
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "lower_fence": lower,
        "upper_fence": upper,
        "outliers": [v for v in values if v < lower or v > upper],  # input order
    }


def iqr_outliers(values: list[float], k: float = 1.5) -> dict:
    """Return the IQR, the lower and upper fences, and the values outside them."""
    q1, _q2, q3 = quartiles(values)
    return _outliers_from_quartiles(values, q1, q3, k)

def _fmt(x: float) -> str:
    return f"{x:,.2f}"


def _fmt_mode(modes: list[float]) -> str:
    return ", ".join(_fmt(m) for m in modes) if modes else "none"


class StatisticsAnalyzer:
    """Wraps one numeric column and exposes every statistic above as a method.

    The data is sorted ONCE in __init__; every result is cached after its
    first calculation, so repeated calls do no extra work.
    """

    def __init__(self, values: list[float], name: str = "values", labels=None,
                 rows_read: int | None = None, rejected=None) -> None:
        if len(values) == 0:
            raise ValueError("StatisticsAnalyzer needs at least one value")
        if labels is not None and len(labels) != len(values):
            raise ValueError("labels must be the same length as values")
        self.name = name
        self.values = list(values)
        self.labels = list(labels) if labels is not None else [None] * len(values)
        self.rows_read = rows_read if rows_read is not None else len(values)
        self.rejected = list(rejected) if rejected else []
        self._sorted = sorted(self.values)
        self._cache: dict = {}

    @classmethod
    def from_csv(cls, path: Path, column: str) -> "StatisticsAnalyzer":
        valid, rejected, rows_read = _scan(Path(path), column)
        return cls([v for v, _ in valid], name=column, labels=[lab for _, lab in valid],
                   rows_read=rows_read, rejected=rejected)

    
    def _cached(self, key, compute):
        if key not in self._cache:
            self._cache[key] = compute()
        return self._cache[key]

    def count(self) -> int:
        return len(self.values)

    def mean(self) -> float:
        return self._cached("mean", lambda: mean(self.values))

    def median(self) -> float:
        return self._cached("median", lambda: _median_sorted(self._sorted))

    def mode(self) -> list[float]:
        return self._cached("mode", lambda: mode(self.values))

    def variance(self, sample: bool = True) -> float:
        return self._cached(("var", sample), lambda: variance(self.values, sample))

    def std_dev(self, sample: bool = True) -> float:
        return self._cached(("std", sample), lambda: math.sqrt(self.variance(sample)))

    def quartiles(self) -> tuple[float, float, float]:
        return self._cached("quartiles", lambda: _quartiles_sorted(self._sorted))

    def outliers(self, k: float = 1.5) -> dict:
        def compute():
            q1, _q2, q3 = self.quartiles()
            return _outliers_from_quartiles(self.values, q1, q3, k)
        return self._cached(("outliers", k), compute)

    def outlier_records(self, k: float = 1.5) -> list[tuple[float, object]]:
        """[(value, label)] for each outlier, in original (file) order."""
        info = self.outliers(k)
        lo, hi = info["lower_fence"], info["upper_fence"]
        return [(v, lab) for v, lab in zip(self.values, self.labels) if v < lo or v > hi]

    
    def summary(self) -> dict:
        q1, q2, q3 = self.quartiles()
        info = self.outliers()
        return {
            "count": self.count(),
            "mean": self.mean(),
            "median": self.median(),
            "mode": self.mode(),
            "variance": self.variance(),
            "std_dev": self.std_dev(),
            "min": self._sorted[0],
            "max": self._sorted[-1],
            "q1": q1,
            "q2": q2,
            "q3": q3,
            "iqr": info["iqr"],
            "lower_fence": info["lower_fence"],
            "upper_fence": info["upper_fence"],
            "outliers": info["outliers"],
        }

    def without_outliers(self) -> "StatisticsAnalyzer":
        """A NEW analyzer without the IQR outliers (one pass; self is untouched)."""
        info = self.outliers()
        lo, hi = info["lower_fence"], info["upper_fence"]
        kept = [(v, lab) for v, lab in zip(self.values, self.labels) if lo <= v <= hi]
        return StatisticsAnalyzer([v for v, _ in kept], name=self.name,
                                  labels=[lab for _, lab in kept],
                                  rows_read=self.rows_read, rejected=self.rejected)

    def report(self) -> str:
        a, c = self.summary(), self.without_outliers().summary()
        L = []
        L.append(f"=== Statistics report: {self.name} ===")
        L.append(f"{'Rows read':<18}: {self.rows_read}")
        L.append(f"{'Valid values':<18}: {self.count()}")
        L.append(f"{'Rejected values':<18}: {len(self.rejected)}")
        for row_number, raw in self.rejected:
            L.append(f"  row {row_number}: {raw!r} ({explain_rejection(raw)})")
        L.append("")

        def pair(x, y):
            return f"{_fmt(x)} / {_fmt(y)}"

        table = [
            ("count", str(a["count"]), str(c["count"])),
            ("mean", _fmt(a["mean"]), _fmt(c["mean"])),
            ("median", _fmt(a["median"]), _fmt(c["median"])),
            ("mode", _fmt_mode(a["mode"]), _fmt_mode(c["mode"])),
            ("variance (sample)", _fmt(a["variance"]), _fmt(c["variance"])),
            ("std dev (sample)", _fmt(a["std_dev"]), _fmt(c["std_dev"])),
            ("min / max", pair(a["min"], a["max"]), pair(c["min"], c["max"])),
            ("Q1 / Q3", pair(a["q1"], a["q3"]), pair(c["q1"], c["q3"])),
            ("IQR", _fmt(a["iqr"]), _fmt(c["iqr"])),
        ]
        w = max(len(r[1]) for r in table + [("", "all data", "")]) + 4
        L.append(" " * 20 + "all data".ljust(w) + "outliers removed")
        for label, x, y in table:
            L.append(f"{label:<20}{x.ljust(w)}{y}")
        L.append("")
        L.append(f"{'Outlier fences':<20}: lower {_fmt(a['lower_fence'])}, upper {_fmt(a['upper_fence'])}")
        recs = self.outlier_records()
        if recs:
            text = ", ".join(f"{_fmt(v)} ({lab[0]}, {lab[1]})" if lab else _fmt(v) for v, lab in recs)
        else:
            text = "none"
        L.append(f"{'Outliers':<20}: {text}")
        return "\n".join(L)



def compare_with_numpy(analyzer: StatisticsAnalyzer) -> dict[str, bool]:
    """Recompute each statistic with NumPy and return {name: matches}.

    Uses ddof=1 (sample) and quartile method="linear", the same choices as the
    hand-written code, and compares with a tolerance instead of ==.
    """
    import numpy as np

    data = np.array(analyzer.values, dtype=float)
    q1, _q2, q3 = analyzer.quartiles()
    np_q1, np_q3 = np.percentile(data, [25, 75], method=QUARTILE_METHOD)

    def close(mine, theirs):
        return bool(np.isclose(mine, theirs, rtol=TOLERANCE, atol=TOLERANCE))

    return {
        "mean": close(analyzer.mean(), np.mean(data)),
        "median": close(analyzer.median(), np.median(data)),
        "variance": close(analyzer.variance(), np.var(data, ddof=NUMPY_DDOF)),
        "std_dev": close(analyzer.std_dev(), np.std(data, ddof=NUMPY_DDOF)),
        "quartiles": close(q1, np_q1) and close(q3, np_q3),
    }


MISMATCH_HINTS = {
    "mean": "check that both sides use the same values",
    "median": "check that both sides use the same values",
    "variance": "NumPy defaults to ddof=0 (population); pass ddof=1 for sample variance",
    "std_dev": "NumPy defaults to ddof=0 (population); pass ddof=1 for sample std dev",
    "quartiles": "quartile methods differ; NumPy's method= must be 'linear' to match",
}


def numpy_section(analyzer: StatisticsAnalyzer) -> str:
    results = compare_with_numpy(analyzer)
    flag = lambda ok: "MATCH" if ok else "MISMATCH"
    L = ["=== NumPy verification ==="]
    L.append(f"{'mean':<10}{flag(results['mean'])}")
    L.append(f"{'median':<10}{flag(results['median'])}")
    L.append(f"{'variance':<10}{flag(results['variance'])}  (ddof={NUMPY_DDOF})")
    L.append(f"{'std dev':<10}{flag(results['std_dev'])}")
    L.append(f"{'Q1 / Q3':<10}{flag(results['quartiles'])}  (method={QUARTILE_METHOD})")
    for key, ok in results.items():
        if not ok:
            L.append(f"  explanation ({key}): {MISMATCH_HINTS[key]}")
    return "\n".join(L)



def risk_notes(analyzer: StatisticsAnalyzer) -> str:
    """Three sentences for the risk team, built from the actual outliers."""
    recs = analyzer.outlier_records()
    if not recs:
        return ("No day fell outside the IQR fences, so no single deposit total looks extreme. "
                "Rejected rows still need follow-up because they hide days we cannot judge. "
                "Re-run this report as new data arrives.")
    info = analyzer.outliers()
    high = [r for r in recs if r[0] > info["upper_fence"]]
    low = [r for r in recs if r[0] < info["lower_fence"]]

    def describe(group):
        return ", ".join(f"{_fmt(v)} on {lab[0]} by {lab[1]}" if lab else _fmt(v) for v, lab in group)

    parts = []
    if high:
        parts.append(f"{len(high)} unusually high ({describe(high)})")
    if low:
        parts.append(f"{len(low)} unusually low ({describe(low)})")
    agents = sorted({lab[1] for _v, lab in recs if lab})
    who = f"all come from agent {agents[0]}" if len(agents) == 1 and len(recs) > 1 else (
        f"come from agents {', '.join(agents)}" if agents else "")
    s1 = (f"The IQR rule flags {len(recs)} of {analyzer.count()} valid days: "
          + " and ".join(parts) + (f", and they {who}." if who else "."))
    s2 = ("Very high totals can mean fraud or structuring, a bulk float top-up logged as a deposit, "
          "or an extra digit typed by the agent, while very low totals can mean a float shortage, "
          "a network outage or a missing digit.")
    cleaned = analyzer.without_outliers()
    s3 = (f"None were deleted from the source data; they only move the mean from "
          f"{_fmt(analyzer.mean())} to {_fmt(cleaned.mean())}, so each should be checked against "
          f"the M-Pesa statement for that agent and day before being called genuine or an error.")
    return " ".join([s1, s2, s3])


def run_unit_tests() -> str:
    proc = subprocess.run([sys.executable, "-m", "unittest", "test_stats_toolkit"],
                          cwd=BASE_DIR, capture_output=True, text=True)
    lines = [ln for ln in proc.stderr.strip().splitlines() if ln.strip()]
    ran = next((ln for ln in lines if ln.startswith("Ran ")), "Ran 0 tests")
    return f"{ran}\n{lines[-1] if lines else 'FAILED (no output)'}"


def build_full_report(analyzer: StatisticsAnalyzer, with_tests: bool = True) -> str:
    sections = [analyzer.report(), numpy_section(analyzer)]
    sections.append("=== Unit tests ===\n" + (run_unit_tests() if with_tests else "not run"))
    sections.append("=== Notes for the risk team ===\n" + risk_notes(analyzer))
    return "\n\n".join(sections) + "\n"


def main(argv: list[str]) -> None:
    csv_path = Path(argv[1]) if len(argv) > 1 else DATA_PATH
    analyzer = StatisticsAnalyzer.from_csv(csv_path, COLUMN)
    text = build_full_report(analyzer)
    print(text)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(text, encoding="utf-8")
    print(f"(saved to {REPORT_PATH})")


if __name__ == "__main__":
    main(sys.argv)
