import numpy as np
import pandas as pd
import pytest

from app import fundamentals as f
from app.data import DataUnavailableError
from app.indicators import build_indicators, moving_averages, rsi_indicator, volumes
from app.instruments import Instrument
from tests.conftest import make_fundamentals, make_prices

MA = Instrument("IAM", "Maroc Telecom", 700, None, None)
STOCK = Instrument("STMPA.PA", "STMicro", 1280, None, None, market="INTL", currency="EUR")
ETF = Instrument("CW8.PA", "MSCI World", 1280, None, None, market="INTL", currency="EUR", kind="ETF")


def group(result, group_id):
    return next(g for g in result["groups"] if g["id"] == group_id)


def test_requested_indicators_exact_set_and_order():
    result = build_indicators(STOCK, make_prices(), lambda s: make_fundamentals())
    assert [g["id"] for g in result["groups"]] == ["fundamental", "technical"]
    labels = [i["label"] for g in result["groups"] for i in g["items"]]
    assert labels == [
        "PER",
        "PEG",
        "Free Cash Flow (FCF)",
        "ROE / ROIC",
        "Dette nette / EBITDA",
        "Marge opérationnelle",
        "Payout Ratio",
        "Moyennes Mobiles (MM50 / MM200)",
        "RSI",
        "Volumes",
    ]


def test_moroccan_stock_has_technicals_but_no_fundamentals():
    called = []
    result = build_indicators(MA, make_prices(), lambda s: called.append(s))
    assert called == []  # aucune requête de fondamentaux pour le Maroc
    fundamentals = group(result, "fundamental")
    assert fundamentals["status"] == f.UNAVAILABLE and "PDF" in fundamentals["note"]
    assert all(v["status"] == f.UNAVAILABLE and v["value"] is None for i in fundamentals["items"] for v in i["values"])
    assert all(v["status"] == f.OK for i in group(result, "technical")["items"][:2] for v in i["values"])


def test_etf_fundamentals_not_applicable():
    called = []
    result = build_indicators(ETF, make_prices(), lambda s: called.append(s))
    assert called == []
    fundamentals = group(result, "fundamental")
    assert fundamentals["status"] == f.NOT_APPLICABLE and "ETF" in fundamentals["note"]
    assert all(v["status"] == f.NOT_APPLICABLE for i in fundamentals["items"] for v in i["values"])
    assert len(fundamentals["items"]) == 7


def test_source_outage_degrades_gracefully():
    def boom(symbol):
        raise DataUnavailableError("panne")

    result = build_indicators(STOCK, make_prices(), boom)
    assert group(result, "fundamental")["status"] == f.UNAVAILABLE
    assert "momentanément" in group(result, "fundamental")["note"]
    assert group(result, "technical")["status"] == "ok"  # le technique reste affiché


def test_company_without_published_data():
    result = build_indicators(STOCK, make_prices(), lambda s: {"annualEBITDA": []})
    assert "Aucune donnée" in group(result, "fundamental")["note"]


def test_moving_averages_values_and_reading():
    close = pd.Series(np.arange(1.0, 301.0))
    item = moving_averages(close, "EUR")
    v = {x["label"]: x for x in item["values"]}
    assert v["MM50"]["value"] == pytest.approx(close.tail(50).mean(), abs=1e-4)
    assert v["MM200"]["value"] == pytest.approx(close.tail(200).mean(), abs=1e-4)
    assert v["Cours vs MM50"]["value"] == pytest.approx(300 / close.tail(50).mean() - 1, abs=1e-4)
    assert "haussière" in item["reading"]
    assert "baissière" in moving_averages(close[::-1].reset_index(drop=True), "EUR")["reading"]


def test_moving_average_200_unavailable_on_short_history():
    item = moving_averages(pd.Series(np.arange(1.0, 99.0)), "USD")
    v = {x["label"]: x for x in item["values"]}
    assert v["MM50"]["status"] == f.OK
    assert v["MM200"]["status"] == f.UNAVAILABLE and "98" in v["MM200"]["note"]
    assert item["reading"] is None


def test_rsi_zones():
    up = pd.Series(np.arange(1.0, 100.0))
    down = up[::-1].reset_index(drop=True)
    assert rsi_indicator(up)["values"][0]["value"] > 70 and "Surachat" in rsi_indicator(up)["reading"]
    assert rsi_indicator(down)["values"][0]["value"] < 30 and "Survente" in rsi_indicator(down)["reading"]
    assert rsi_indicator(pd.Series([1.0, 2.0]))["values"][0]["status"] == f.UNAVAILABLE


def test_volumes_reading_and_missing():
    quiet = pd.Series([100.0] * 60)
    busy = pd.Series([100.0] * 55 + [300.0] * 5)
    assert "proche" in volumes(quiet)["reading"]
    item = volumes(busy)
    assert "supérieur" in item["reading"]
    assert {v["label"]: v["value"] for v in item["values"]}["Dernière séance"] == 300.0
    assert volumes(pd.Series([0.0] * 60))["values"][0]["status"] == f.UNAVAILABLE


def test_json_safe_values():
    result = build_indicators(STOCK, make_prices(), lambda s: make_fundamentals())
    for g in result["groups"]:
        for i in g["items"]:
            for v in i["values"]:
                assert v["value"] is None or np.isfinite(v["value"])
