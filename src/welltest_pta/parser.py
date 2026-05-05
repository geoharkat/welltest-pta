"""
welltest_pta.parser
===================
Robust multi-format ASCII reader for well-test gauge data.

Handles arbitrary delimiters (tab, semicolon, comma, pipe, whitespace),
mixed encodings, comma-decimal European format, and various date formats
(DD/MM/YYYY HH:MM:SS, MM/DD/YY HH:MM, etc.).

Refactored from the original ``EnhancedWellTestParser`` (Harkat 2025) into
a stateless function plus a stateful class for advanced use cases.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Module-level parsing constants
# ─────────────────────────────────────────────────────────────────────────────

P_RANGE_DEFAULT = (0.0, 30000.0)
T_RANGE_DEFAULT = (-50.0, 500.0)
YEAR_RANGE_DEFAULT = (1980, 2035)

P_KEYWORDS = [
    "bhp", "press", "pression", "psia", "psig",
    "p-avg", "gauge", "pressure", "pres",
]
T_KEYWORDS = [
    "bht", "temp", "degc", "degf", "t-avg",
    "temperature", "deg", "°c", "°f",
]
DT_KEYWORDS = [
    "date", "time", "temps", "hh:mm", "hh/mm", "clock",
    "yyyy", "timestamp", "datetime", "dd/mm", "mm/dd", "dd-", "yy",
]
DELTA_KEYWORDS = ["delta", "elapsed", "cumul"]
UNIT_INDICATORS = [
    "psia", "psig", "degc", "degf", "°c", "°f",
    "hh:mm", "hh/mm", "mm/dd", "dd/mm",
    "mpa", "kpa", "bar", "atm",
]
META_PATTERNS = [
    r"well\s*name", r"gauge\s*(serial|model|manufacturer|s/n)",
    r"client", r"field", r"rig\s*name", r"date\s*of\s*last",
    r"pressure\s*units", r"temperature\s*units", r"type\s*de\s*test",
    r"d[ée]but", r"fin\s*des", r"intervalle",
    r"point\s*de\s*lecture", r"c[ôo]te",
    r"gauge\s*spe[sc]ialist", r"=====",
]


# ─────────────────────────────────────────────────────────────────────────────
# Public top-level function
# ─────────────────────────────────────────────────────────────────────────────

def parse(
    filepath: str | Path,
    p_range: tuple[float, float] = P_RANGE_DEFAULT,
    t_range: tuple[float, float] = T_RANGE_DEFAULT,
) -> pd.DataFrame:
    """
    Parse a well-test ASCII gauge file into a clean DataFrame.

    Parameters
    ----------
    filepath
        Path to the ASCII (.txt, .csv, .dat, .prn, ...) file.
    p_range
        Physical sanity bounds for pressure (psia). Values outside are NaN-ed.
    t_range
        Physical sanity bounds for temperature.

    Returns
    -------
    DataFrame with at least ``timestamp`` and ``pressure`` columns, plus
    ``temperature``, ``delta_hours``, and QC flags when available.
    """
    parser = WellTestParser(p_range=p_range, t_range=t_range)
    return parser.parse(filepath)


# ─────────────────────────────────────────────────────────────────────────────
# Stateful parser class
# ─────────────────────────────────────────────────────────────────────────────

class WellTestParser:
    """
    Stateful parser that exposes detected metadata and column mapping after
    a successful ``.parse(...)`` call.
    """

    def __init__(
        self,
        p_range: tuple[float, float] = P_RANGE_DEFAULT,
        t_range: tuple[float, float] = T_RANGE_DEFAULT,
        year_range: tuple[int, int] = YEAR_RANGE_DEFAULT,
    ) -> None:
        self.P_RANGE = p_range
        self.T_RANGE = t_range
        self.YEAR_RANGE = year_range
        self.metadata: dict[str, str] = {}
        self.mapping: dict[str, Any] = {}
        self._comma_decimal: bool = False
        self._dayfirst_hint: Optional[bool] = None

    # ── Numeric and time-string helpers ──────────────────────────────────

    def _clean_numeric(self, series: pd.Series) -> pd.Series:
        def extract(val):
            if pd.isna(val) or str(val).strip() == "":
                return np.nan
            s = str(val).strip()
            if self._comma_decimal:
                s = re.sub(r"(\d),(\d)", r"\1.\2", s)
            try:
                m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
                return float(m.group()) if m else np.nan
            except (ValueError, TypeError):
                return np.nan
        return series.apply(extract)

    @staticmethod
    def _normalize_time_string(val, is_time_col: bool = True):
        if pd.isna(val):
            return val
        s = str(val).strip()
        if not is_time_col:
            return s
        m = re.match(r"^(\d{1,2})/(\d{1,2})(/(\d{1,2}))?$", s)
        if m:
            hh, mm = int(m.group(1)), int(m.group(2))
            if hh <= 23 and mm <= 59:
                return s.replace("/", ":")
        return s

    @staticmethod
    def _is_metadata_line(line: str) -> bool:
        stripped = line.strip()
        if not stripped:
            return True
        if re.match(r"^[=\-_\*#~]{3,}$", stripped):
            return True
        low = stripped.lower()
        return any(re.search(p, low) for p in META_PATTERNS)

    @staticmethod
    def _extract_metadata(lines: list[str]) -> dict[str, str]:
        meta: dict[str, str] = {}
        for line in lines:
            stripped = line.strip()
            if not stripped or re.match(r"^[=\-_\*#~]{3,}$", stripped):
                continue
            m = re.match(r"^(.+?)\s*[:=]\s*(.+)$", stripped)
            if m:
                meta[m.group(1).strip()] = m.group(2).strip()
        return meta

    @staticmethod
    def _detect_comma_decimal(lines: list[str], start: int) -> bool:
        cc = dc = checked = 0
        for line in lines[start:start + 30]:
            s = line.strip()
            if not s:
                continue
            cc += len(re.findall(r"\d+,\d{1,5}(?!\d)", s))
            dc += len(re.findall(r"\d+\.\d{1,5}(?!\d)", s))
            checked += 1
            if checked >= 10:
                break
        return cc > dc and cc > 3

    @staticmethod
    def _detect_dayfirst_from_headers(header_line: str) -> Optional[bool]:
        low = header_line.lower()
        if re.search(r"dd[/\-]mm", low):
            return True
        if re.search(r"mm[/\-]dd", low):
            return False
        return None

    # ── Structure detection ──────────────────────────────────────────────

    def _detect_structure(
        self, filepath: Path, max_lines: int = 300
    ) -> tuple[int, str, Optional[int], list[str]]:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            lines = [line.rstrip("\n") for line in f][:max_lines]

        meta_lines: list[str] = []
        data_start_zone = 0
        for i, line in enumerate(lines):
            if self._is_metadata_line(line):
                meta_lines.append(line)
                data_start_zone = i + 1
            else:
                break

        delimiters = ["\t", ";", ",", "|", r"\s+"]
        all_kw = P_KEYWORDS + T_KEYWORDS + DT_KEYWORDS
        best_score = -1
        best_delim: Optional[str] = None
        best_header = data_start_zone
        best_units: Optional[int] = None

        search_start = max(0, data_start_zone - 3)
        search_end = min(len(lines), data_start_zone + 20)

        for delim in delimiters:
            for i in range(search_start, search_end):
                line = lines[i]
                stripped = line.strip()
                if not stripped or re.match(r"^[=\-_\*#~]{3,}$", stripped):
                    continue
                parts = stripped.split() if delim == r"\s+" else stripped.split(delim)
                if len(parts) < 2:
                    continue
                line_lower = stripped.lower()
                kw_matches = sum(1 for kw in all_kw if kw in line_lower)
                if kw_matches < 1:
                    continue

                consistent, n_fields, data_lines_found = True, None, 0
                for j in range(i + 1, min(i + 15, len(lines))):
                    test = lines[j].strip()
                    if not test:
                        continue
                    if any(u in test.lower() for u in UNIT_INDICATORS):
                        continue
                    if re.match(r"^[=\-_\*#~]{3,}$", test):
                        continue
                    tparts = test.split() if delim == r"\s+" else test.split(delim)
                    if n_fields is None:
                        n_fields = len(tparts)
                    elif abs(len(tparts) - n_fields) > 1:
                        consistent = False
                        break
                    data_lines_found += 1
                    if data_lines_found >= 5:
                        break

                units_idx: Optional[int] = None
                if i + 1 < len(lines):
                    next_low = lines[i + 1].lower().strip()
                    unit_hits = sum(1 for u in UNIT_INDICATORS if u in next_low)
                    if unit_hits >= 1 or re.search(r"(dd|mm|yy|hh)", next_low):
                        if not re.findall(r"\d{4,}", next_low):
                            units_idx = i + 1

                score = (
                    kw_matches * 3
                    + (5 if units_idx else 0)
                    + (15 if consistent else 0)
                    + (3 if data_lines_found >= 3 else 0)
                )
                if score > best_score:
                    best_score = score
                    best_delim = delim
                    best_header = i
                    best_units = units_idx

        if best_delim is None:
            best_delim, best_header, best_units = r"\s+", 0, None
        return best_header, best_delim, best_units, meta_lines

    # ── Column scoring & identification ──────────────────────────────────

    def _score_column(
        self,
        series: pd.Series,
        keywords: list[str],
        phys_range: tuple[float, float],
    ) -> float:
        clean = self._clean_numeric(series).dropna()
        if len(clean) == 0:
            return 0.0
        range_frac = clean.between(*phys_range).mean()
        col_name = str(series.name).lower() if series.name else ""
        col_clean = re.sub(r"[\[\]\(\)\{\}]", " ", col_name)
        name_score = 0.0
        for kw in keywords:
            if re.search(r"\b" + re.escape(kw) + r"\b", col_clean):
                name_score = 3.0
                break
            if kw in col_clean:
                name_score = max(name_score, 1.5)
        return range_frac * (1.0 + name_score)

    def _identify_columns(self, df: pd.DataFrame) -> dict[str, Any]:
        dt_candidates: list[str] = []
        delta_col: Optional[str] = None
        for col in df.columns:
            col_lower = str(col).lower()
            col_clean = re.sub(r"[\[\]\(\)\{\}]", " ", col_lower)
            if any(kw in col_clean for kw in DELTA_KEYWORDS):
                delta_col = col
                continue
            is_dt = any(kw in col_clean for kw in DT_KEYWORDS)
            if not is_dt:
                sample = df[col].dropna().head(5).astype(str)
                for val in sample:
                    if re.search(
                        r"\d{1,4}[/\-]\d{1,2}[/\-]\d{1,4}|\d{1,2}:\d{2}|"
                        r"\d{1,2}/\d{2}(/\d{2})?|\w{3}-\d{2}",
                        val,
                    ):
                        is_dt = True
                        break
            if is_dt:
                dt_candidates.append(col)

        exclude = set(dt_candidates)
        if delta_col:
            exclude.add(delta_col)
        p_scores: dict[str, float] = {}
        t_scores: dict[str, float] = {}
        for col in df.columns:
            if col in exclude:
                continue
            p_scores[col] = self._score_column(df[col], P_KEYWORDS, self.P_RANGE)
            t_scores[col] = self._score_column(df[col], T_KEYWORDS, self.T_RANGE)
        p_col = (
            max(p_scores, key=p_scores.get)
            if p_scores and max(p_scores.values()) > 0.3 else None
        )
        t_col = (
            max(t_scores, key=t_scores.get)
            if t_scores and max(t_scores.values()) > 0.3 else None
        )
        if p_col and p_col == t_col:
            if p_scores.get(p_col, 0) >= t_scores.get(t_col, 0):
                t_col = None
            else:
                p_col = None
        return {"p": p_col, "t": t_col, "dt": dt_candidates, "delta": delta_col}

    # ── Datetime parsing with day/month-first auto-detection ─────────────

    def _parse_datetime(
        self, df: pd.DataFrame, dt_cols: list[str]
    ) -> Optional[pd.Series]:
        if not dt_cols:
            return None

        def looks_like_date(v: str) -> bool:
            m = re.match(r"^(\d{1,4})[/\-](\d{1,2})[/\-](\d{1,4})$", v.strip())
            if not m:
                return False
            a, b = int(m.group(1)), int(m.group(2))
            if len(m.group(1)) == 4 or len(m.group(3)) == 4:
                return True
            if b <= 12:
                return True
            if a <= 12 and b <= 31:
                return True
            return False

        for col in dt_cols:
            sample = df[col].dropna().head(5).astype(str).tolist()
            has_colons = any(":" in v for v in sample)
            date_votes = sum(1 for v in sample if looks_like_date(v))
            is_pure_time = has_colons or (date_votes == 0)
            df[col] = df[col].apply(
                lambda v, _is_t=is_pure_time: self._normalize_time_string(v, is_time_col=_is_t)
            )

        combined = df[dt_cols].astype(str).agg(" ".join, axis=1)
        combined = combined.str.replace(r"\s+", " ", regex=True).str.strip()

        if self._dayfirst_hint is None:
            dayfirst_options = [True, False]
        else:
            dayfirst_options = [self._dayfirst_hint, not self._dayfirst_hint]

        best_series, best_score = None, -1.0
        for dayfirst in dayfirst_options:
            try:
                ts = pd.to_datetime(
                    combined, dayfirst=dayfirst, errors="coerce", format="mixed"
                )
            except Exception:
                continue
            valid_frac = ts.notna().mean()
            if valid_frac < 0.3:
                continue
            years = ts.dt.year
            year_frac = (
                (years >= self.YEAR_RANGE[0]) & (years <= self.YEAR_RANGE[1])
            ).mean()
            diffs = ts.diff().dt.total_seconds().dropna()
            mono_frac = (diffs >= 0).mean() if len(diffs) > 0 else 0
            hint_bonus = 0.1 if dayfirst == self._dayfirst_hint else 0.0
            score = (
                valid_frac * 0.3 + year_frac * 0.3
                + mono_frac * 0.3 + hint_bonus
            )
            if score > best_score:
                best_score = score
                best_series = ts

        return best_series

    # ── QC flags ─────────────────────────────────────────────────────────

    def _add_qc_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        qc = pd.DataFrame(index=df.index)
        qc["qc_pressure"] = "PASS"
        qc["qc_temperature"] = "PASS"
        qc["qc_timestamp"] = "PASS"
        if "pressure" in df.columns:
            p = df["pressure"]
            qc.loc[~p.between(*self.P_RANGE) | p.isna(), "qc_pressure"] = "OUT_OF_RANGE"
        if "temperature" in df.columns:
            t = df["temperature"]
            qc.loc[~t.between(*self.T_RANGE) | t.isna(), "qc_temperature"] = "OUT_OF_RANGE"
        if "timestamp" in df.columns:
            qc.loc[df["timestamp"].isna(), "qc_timestamp"] = "MISSING"
            dup = df["timestamp"].duplicated(keep="first")
            qc.loc[dup, "qc_timestamp"] = "DUPLICATE"
            if not df["timestamp"].dropna().empty:
                non_mono = df["timestamp"].diff().dt.total_seconds().fillna(0) < 0
                qc.loc[non_mono, "qc_timestamp"] = "NON_MONOTONIC"
        return qc

    # ── PUBLIC API ───────────────────────────────────────────────────────

    def parse(self, filepath: str | Path) -> pd.DataFrame:
        """Parse the ASCII file and return a clean DataFrame."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            all_lines = [line.rstrip("\n") for line in f]

        header_row, delimiter, units_row, meta_lines = self._detect_structure(filepath)
        self.metadata = self._extract_metadata(meta_lines)

        data_start = header_row + 1
        if units_row is not None:
            data_start = max(data_start, units_row + 1)

        self._comma_decimal = self._detect_comma_decimal(all_lines, data_start)

        self._dayfirst_hint = None
        if header_row < len(all_lines):
            self._dayfirst_hint = self._detect_dayfirst_from_headers(all_lines[header_row])
        if self._dayfirst_hint is None and units_row is not None and units_row < len(all_lines):
            self._dayfirst_hint = self._detect_dayfirst_from_headers(all_lines[units_row])

        skip_rows = list(range(0, header_row))
        if units_row is not None:
            skip_rows.append(units_row)

        try:
            kwargs: dict = dict(
                sep=delimiter, skiprows=skip_rows, engine="python",
                on_bad_lines="skip", encoding="utf-8", encoding_errors="ignore",
            )
            if self._comma_decimal:
                kwargs["dtype"] = str
            df = pd.read_csv(filepath, **kwargs)
        except Exception:
            df = pd.read_csv(
                filepath, sep=r"\s+", header=None, engine="python",
                on_bad_lines="skip", encoding="utf-8", encoding_errors="ignore",
                dtype=(str if self._comma_decimal else None),
            )
            df.columns = [f"col_{i}" for i in range(df.shape[1])]

        df.columns = [str(c).strip() for c in df.columns]

        # Recover unnamed columns by re-reading the raw header
        unnamed_cols = [c for c in df.columns if c.startswith("Unnamed")]
        if unnamed_cols:
            df = self._recover_unnamed_columns(df, all_lines, header_row, units_row)

        df = df.dropna(how="all").dropna(axis=1, how="all")

        # Strip junk rows
        all_kw = P_KEYWORDS + T_KEYWORDS + DT_KEYWORDS

        def is_junk_row(row: pd.Series) -> bool:
            s = " ".join(str(v) for v in row.values if pd.notna(v)).lower()
            if re.match(r"^[=\-_\*#~\s]+$", s):
                return True
            if any(u in s for u in UNIT_INDICATORS):
                return True
            return False

        mask = df.apply(is_junk_row, axis=1)
        if mask.any():
            df = df[~mask].reset_index(drop=True)

        col_roles = self._identify_columns(df)
        self.mapping = col_roles

        timestamp_series = (
            self._parse_datetime(df, col_roles["dt"]) if col_roles["dt"] else None
        )

        pressure_series = None
        if col_roles["p"]:
            pressure_series = self._clean_numeric(df[col_roles["p"]])
            pressure_series = pressure_series.where(pressure_series.between(*self.P_RANGE))

        temperature_series = None
        if col_roles["t"]:
            temperature_series = self._clean_numeric(df[col_roles["t"]])
            temperature_series = temperature_series.where(temperature_series.between(*self.T_RANGE))

        out = pd.DataFrame(index=df.index)
        if timestamp_series is not None:
            out["timestamp"] = timestamp_series
        if pressure_series is not None:
            out["pressure"] = pressure_series
        if temperature_series is not None:
            out["temperature"] = temperature_series
        if col_roles["delta"]:
            out["delta_hours"] = self._clean_numeric(df[col_roles["delta"]])

        used = {col_roles["p"], col_roles["t"], col_roles["delta"]}
        used.update(col_roles["dt"])
        used.discard(None)
        for c in df.columns:
            if c not in used:
                out[f"aux_{c}"] = df[c]

        qc = self._add_qc_flags(out)
        out = pd.concat([out, qc], axis=1)
        if "timestamp" in out.columns:
            out = out.sort_values("timestamp").reset_index(drop=True)
        else:
            out = out.reset_index(drop=True)
        return out

    def _recover_unnamed_columns(
        self,
        df: pd.DataFrame,
        all_lines: list[str],
        header_row: int,
        units_row: Optional[int],
    ) -> pd.DataFrame:
        raw_header = all_lines[header_row] if header_row < len(all_lines) else ""
        real_names = [tok.strip() for tok in re.split(r"\t+|\s{2,}", raw_header) if tok.strip()]
        data_start_line = header_row + 1
        if units_row is not None and units_row >= data_start_line:
            data_start_line = units_row + 1
        all_kw = P_KEYWORDS + T_KEYWORDS + DT_KEYWORDS
        data_rows: list[list[str]] = []
        for line in all_lines[data_start_line:]:
            stripped = line.strip()
            if not stripped:
                continue
            if re.match(r"^[=\-_\*#~]{3,}$", stripped):
                continue
            low = stripped.lower()
            if any(kw in low for kw in all_kw):
                continue
            if any(u in low for u in UNIT_INDICATORS):
                continue
            fields = stripped.split()
            if len(fields) >= 2:
                data_rows.append(fields)
        if not data_rows:
            return df
        n_data, n_names = len(data_rows[0]), len(real_names)
        if n_data > n_names:
            extra = n_data - n_names
            expanded = [real_names[0]]
            for ei in range(extra):
                expanded.append(f"_dt_part_{ei}")
            expanded.extend(real_names[1:])
            real_names = expanded
        uniform: list[list] = []
        for row in data_rows:
            if len(row) >= len(real_names):
                uniform.append(row[:len(real_names)])
            else:
                uniform.append(row + [np.nan] * (len(real_names) - len(row)))
        return pd.DataFrame(uniform, columns=real_names[:len(uniform[0])])
