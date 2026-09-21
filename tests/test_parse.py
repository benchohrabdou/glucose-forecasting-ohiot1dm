"""Parser tests on a small synthetic XML (no real patient data)."""
import pandas as pd

from src.data.parse import load_patient

XML = """<patient id="999" weight="70" insulin_type="Novalog">
  <glucose_level>
    <event ts="07-12-2021 01:17:03" value="101"/>
    <event ts="07-12-2021 01:22:00" value="105"/>
  </glucose_level>
  <basal><event ts="07-12-2021 00:00:00" value="0.9"/></basal>
  <temp_basal><event ts_begin="07-12-2021 01:00:00" ts_end="07-12-2021 02:00:00" value="0"/></temp_basal>
  <bolus>
    <event ts_begin="07-12-2021 01:20:00" ts_end="07-12-2021 01:20:00" type="normal" dose="4.2" bwz_carb_input="45"/>
    <event ts_begin="07-12-2021 02:20:00" ts_end="07-12-2021 02:50:00" type="square" dose="2" bwz_carb_input="NA"/>
  </bolus>
  <meal><event ts="07-12-2021 01:15:00" type="Lunch" carbs="45"/></meal>
  <exercise/>
</patient>
"""


def _load(tmp_path):
    path = tmp_path / "999-ws-training.xml"
    path.write_text(XML)
    return load_patient(path)


def test_meta(tmp_path):
    meta, _ = _load(tmp_path)
    assert meta == {"id": "999", "weight": "70", "insulin_type": "Novalog"}


def test_timestamps_parsed_day_first(tmp_path):
    _, t = _load(tmp_path)
    assert t["glucose_level"]["ts"].iloc[0] == pd.Timestamp("2021-12-07 01:17:03")
    assert pd.api.types.is_datetime64_any_dtype(t["bolus"]["ts_begin"])
    assert pd.api.types.is_datetime64_any_dtype(t["temp_basal"]["ts_end"])


def test_numeric_and_text_columns(tmp_path):
    _, t = _load(tmp_path)
    assert t["glucose_level"]["value"].tolist() == [101, 105]
    assert pd.api.types.is_numeric_dtype(t["bolus"]["dose"])
    assert t["meal"]["type"].tolist() == ["Lunch"]  # text stays text
    assert t["bolus"]["type"].tolist() == ["normal", "square"]


def test_missing_numeric_becomes_nan(tmp_path):
    _, t = _load(tmp_path)
    bwz = t["bolus"]["bwz_carb_input"]
    assert bwz.iloc[0] == 45 and pd.isna(bwz.iloc[1])


def test_empty_field_is_empty_frame(tmp_path):
    _, t = _load(tmp_path)
    assert t["exercise"].empty
