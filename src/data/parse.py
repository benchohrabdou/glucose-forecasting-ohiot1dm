"""Parser for OhioT1DM XML files.

Contract: ``load_patient(path)`` -> ``(meta, tables)`` where ``meta`` is the
``<patient>`` attribute dict and ``tables`` maps each child tag
(``glucose_level``, ``bolus``, ...) to a DataFrame with one row per ``<event>``.

Conventions: timestamps are naive local (de-identified) time, parsed from
``DD-MM-YYYY HH:MM:SS``. Columns named ``ts``, ``ts_begin`` and ``ts_end`` become
datetime64; every other column becomes numeric if all of its non-missing values
are numeric (glucose in mg/dL, basal in U/hr, bolus dose in U, carbs in g) and
stays a string otherwise (e.g. ``type``).
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

TS_FORMAT = "%d-%m-%Y %H:%M:%S"
TS_COLUMNS = ("ts", "ts_begin", "ts_end")
# Placeholders that mean "no value" rather than text content.
MISSING_TOKENS = ("", "NA", "N/A", "NaN", "nan", "null")


def _events_to_frame(element: ET.Element) -> pd.DataFrame:
    """Turn one field element (e.g. ``<glucose_level>``) into a DataFrame."""
    rows = [dict(event.attrib) for event in element.iter("event")]
    frame = pd.DataFrame(rows)
    for col in frame.columns:
        if col in TS_COLUMNS:
            frame[col] = pd.to_datetime(frame[col], format=TS_FORMAT)
        else:
            raw = frame[col].where(~frame[col].isin(MISSING_TOKENS))
            numeric = pd.to_numeric(raw, errors="coerce")
            # Convert only if coercion loses nothing; otherwise the column is
            # categorical text and must not be turned into NaN.
            if numeric.notna().sum() == raw.notna().sum():
                frame[col] = numeric
    return frame


def load_patient(path: str | Path) -> tuple[dict[str, str], dict[str, pd.DataFrame]]:
    """Load one ``<id>-ws-{training,testing}.xml`` file.

    Returns ``(meta, tables)``. Fields with no events (``<exercise/>``) map to an
    empty DataFrame so callers can index any tag without a KeyError. Rows keep
    file order; sorting/deduplication is the preprocessing step's job.
    """
    root = ET.parse(path).getroot()
    meta = dict(root.attrib)
    tables = {child.tag: _events_to_frame(child) for child in root}
    return meta, tables
