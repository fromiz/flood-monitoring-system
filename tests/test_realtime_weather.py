from datetime import datetime, timedelta, timezone

from app.realtime_weather import (
    RealtimeWeatherService,
    parse_apihub_table,
    parse_grid_point_text,
)


def test_parse_apihub_csv_table():
    text = """# START7777
# TM,STN,TA,WS,HM,RN_DAY
202607291600,138,31.2,2.4,61,4.5
202607291601,138,31.1,2.5,62,4.7
#7777END
"""
    rows = parse_apihub_table(text)
    assert len(rows) == 2
    assert rows[-1]["TM"] == "202607291601"
    assert rows[-1]["RN_DAY"] == "4.7"


def test_parse_apihub_whitespace_and_hourly_rows():
    text = """# TM STN TA WS HM RN_DAY
2026072916 138 31.2 2.4 61 4.5
202607291700 138 30.8 2.1 63 4.5
"""
    rows = parse_apihub_table(text)
    assert len(rows) == 2
    assert rows[0]["TA"] == "31.2"
    assert rows[-1]["HM"] == "63"


def test_parse_apihub_yymmddhhmi_csv_header():
    text = """# YYMMDDHHMI,STN,TA,WS1,HM,RN_DAY
202607291601,138,31.1,2.5,62,4.7
"""
    rows = parse_apihub_table(text)
    assert rows == [{
        "TM": "202607291601",
        "STN": "138",
        "TA": "31.1",
        "WS1": "2.5",
        "HM": "62",
        "RN_DAY": "4.7",
    }]


def test_latest_valid_weather_metric_carries_forward_only_recent_value():
    service = RealtimeWeatherService()
    latest_at = datetime(2026, 8, 14, 8, 20, tzinfo=timezone.utc)
    history = [
        {
            "observed_at": (latest_at - timedelta(minutes=3)).isoformat(),
            "temperature_c": 27.4,
        },
        {
            "observed_at": latest_at.isoformat(),
            "temperature_c": None,
        },
    ]
    assert service._latest_valid_metric(
        history,
        "temperature_c",
        latest_at,
    ) == 27.4


def test_parse_grid_point_ascii():
    text = """# TM LON LAT VALUE
202607291550 129.38002 36.03201 3.2
202607291555 129.38002 36.03201 4.1
"""
    observed_at, value = parse_grid_point_text(text)
    assert observed_at is not None
    assert observed_at.minute == 55
    assert value == 4.1
