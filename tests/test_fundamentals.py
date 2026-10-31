from datetime import date

import httpx
import pytest

from app import fundamentals as f
from app.data import DataUnavailableError
from tests.conftest import make_fundamentals

TODAY = date(2026, 10, 30)


def by_id(raw, today=TODAY, currency="EUR"):
    return {i["id"]: i for i in f.compute_fundamentals(raw, today, currency)}


def values(item):
    return {v["label"]: v for v in item["values"]}


def test_healthy_company_all_ratios():
    items = by_id(make_fundamentals(TODAY))
    assert values(items["per"])["PER"]["value"] == 18.5
    assert values(items["peg"])["PEG"]["value"] == 1.4
    fcf = values(items["fcf"])
    assert fcf["FCF"]["value"] == 800.0 and fcf["FCF"]["currency"] == "EUR"
    assert fcf["Marge de FCF"]["value"] == pytest.approx(0.08)  # 800 / 10 000, période TTM la plus récente
    assert "12 derniers mois" in fcf["FCF"]["period"]
    roe_roic = values(items["roe_roic"])
    assert roe_roic["ROE"]["value"] == pytest.approx(1000 / 8000)
    assert roe_roic["ROIC"]["value"] == pytest.approx(1300 * 0.75 / 12000, abs=1e-4)
    assert "25%" in roe_roic["ROIC"]["note"]
    leverage = values(items["net_debt_ebitda"])
    assert leverage["Dette nette / EBITDA"]["value"] == pytest.approx(1.0)  # (3000 - 1000) / 2000
    assert leverage["Dette nette"]["value"] == 2000.0
    assert values(items["operating_margin"])["Marge opérationnelle"]["value"] == pytest.approx(0.15)
    assert values(items["payout"])["Payout Ratio"]["value"] == pytest.approx(0.4)


def test_order_and_ids():
    assert [i["id"] for i in f.compute_fundamentals(make_fundamentals(TODAY), TODAY)] == [
        "per", "peg", "fcf", "roe_roic", "net_debt_ebitda", "operating_margin", "payout",
    ]  # fmt: skip


def test_loss_making_company_is_not_meaningful_not_invented():
    raw = make_fundamentals(TODAY)
    raw["trailingPeRatio"] = [("2024-09-19", 3.2, None)]  # périmé
    raw["trailingNetIncome"] = [("2026-06-30", -500.0, "EUR")]
    raw["annualNetIncome"] = [("2025-12-31", -900.0, "EUR")]
    raw["annualEBITDA"] = [("2025-12-31", -100.0, "EUR")]
    items = by_id(raw)
    assert values(items["per"])["PER"]["status"] == f.NOT_MEANINGFUL
    assert values(items["per"])["PER"]["value"] is None
    assert values(items["payout"])["Payout Ratio"]["status"] == f.NOT_MEANINGFUL
    leverage = values(items["net_debt_ebitda"])
    assert leverage["Dette nette / EBITDA"]["status"] == f.NOT_MEANINGFUL
    assert leverage["Dette nette"]["status"] == f.OK  # le montant reste informatif
    # ROE négatif : une perte sur des capitaux propres positifs est une information, pas un cas « non significatif »
    assert values(items["roe_roic"])["ROE"]["value"] == pytest.approx(-900 / 8000)


def test_stale_per_without_loss_is_unavailable():
    raw = make_fundamentals(TODAY)
    raw["trailingPeRatio"] = [("2024-09-19", 12.0, None)]
    assert values(by_id(raw)["per"])["PER"]["status"] == f.UNAVAILABLE


def test_peg_stale_missing_or_negative():
    raw = make_fundamentals(TODAY)
    raw["trailingPegRatio"] = [("2025-01-01", 1.0, None)]
    assert values(by_id(raw)["peg"])["PEG"]["status"] == f.UNAVAILABLE
    raw["trailingPegRatio"] = [(TODAY.isoformat(), -0.5, None)]
    assert values(by_id(raw)["peg"])["PEG"]["status"] == f.NOT_MEANINGFUL
    del raw["trailingPegRatio"]
    assert values(by_id(raw)["peg"])["PEG"]["status"] == f.UNAVAILABLE


def test_periods_are_never_mixed():
    raw = make_fundamentals(TODAY)
    # Capitaux propres d'un autre exercice que le résultat net : pas de ROE calculé.
    raw["annualStockholdersEquity"] = [("2024-12-31", 8_000.0, "EUR")]
    assert values(by_id(raw)["roe_roic"])["ROE"]["status"] == f.UNAVAILABLE
    # Dividendes d'un autre exercice que le résultat net : payout indisponible.
    raw = make_fundamentals(TODAY)
    raw["annualCashDividendsPaid"] = [("2023-12-31", -400.0, "EUR")]
    assert values(by_id(raw)["payout"])["Payout Ratio"]["status"] == f.UNAVAILABLE


def test_roic_uses_default_tax_when_rate_absurd():
    raw = make_fundamentals(TODAY)
    raw["annualTaxRateForCalcs"] = [("2025-12-31", 3.7, "EUR")]
    roic = values(by_id(raw)["roe_roic"])["ROIC"]
    assert roic["value"] == pytest.approx(1300 * 0.75 / 12000, abs=1e-4) and "supposé" in roic["note"]


def test_net_debt_prefers_reported_value_then_falls_back():
    raw = make_fundamentals(TODAY)
    raw["annualNetDebt"] = [("2025-12-31", 500.0, "EUR")]
    amount = values(by_id(raw)["net_debt_ebitda"])["Dette nette"]
    assert amount["value"] == 500.0 and "publiée" in amount["note"]
    del raw["annualNetDebt"]
    assert values(by_id(raw)["net_debt_ebitda"])["Dette nette"]["value"] == 2000.0
    for key in ("annualTotalDebt", "annualNetDebt"):
        raw.pop(key, None)
    assert values(by_id(raw)["net_debt_ebitda"])["Dette nette / EBITDA"]["status"] == f.UNAVAILABLE


def test_net_cash_is_flagged():
    raw = make_fundamentals(TODAY)
    raw["annualCashCashEquivalentsAndShortTermInvestments"] = [("2025-12-31", 5_000.0, "EUR")]
    ratio = values(by_id(raw)["net_debt_ebitda"])["Dette nette / EBITDA"]
    assert ratio["value"] == pytest.approx(-1.0) and "Trésorerie nette" in ratio["note"]


def test_payout_above_100_percent_flagged():
    raw = make_fundamentals(TODAY)
    raw["annualCashDividendsPaid"] = [("2025-12-31", -1_500.0, "EUR")]
    payout = values(by_id(raw)["payout"])["Payout Ratio"]
    assert payout["value"] == pytest.approx(1.5) and "supérieure" in payout["note"]


def test_empty_raw_everything_unavailable():
    for indicator in f.compute_fundamentals({}, TODAY):
        assert all(v["status"] == f.UNAVAILABLE and v["value"] is None for v in indicator["values"])


def test_annual_fallback_when_no_ttm():
    raw = make_fundamentals(TODAY)
    for key in ("trailingFreeCashFlow", "trailingTotalRevenue", "trailingOperatingIncome"):
        del raw[key]
    items = by_id(raw)
    assert values(items["fcf"])["FCF"]["value"] == 700.0 and "Exercice" in values(items["fcf"])["FCF"]["period"]
    assert values(items["operating_margin"])["Marge opérationnelle"]["value"] == pytest.approx(1300 / 9500, abs=1e-4)


# ---- client Yahoo timeseries -------------------------------------------------------------------


def series(key, points):
    return {"meta": {"symbol": "X", "type": [key]}, key: points}


def payload(*results):
    return {"timeseries": {"result": list(results), "error": None}}


def point(as_of, raw, currency="EUR"):
    return {"asOfDate": as_of, "currencyCode": currency, "reportedValue": {"raw": raw, "fmt": str(raw)}}


def test_fetch_parses_sorts_and_skips_empty_points():
    body = payload(
        series("annualNetIncome", [point("2025-12-31", 5.0), None, point("2024-12-31", 3.0)]),
        series("trailingPeRatio", [{"asOfDate": "2026-10-29", "reportedValue": {"raw": 20.0}}]),
        {"meta": {"symbol": "X", "type": ["annualEBITDA"]}},  # aucune donnée
    )
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url)
        return httpx.Response(200, json=body)

    raw = f.fetch_fundamentals("A/B", client=httpx.Client(transport=httpx.MockTransport(handler)), now=1_800_000_000)
    assert raw["annualNetIncome"] == [("2024-12-31", 3.0, "EUR"), ("2025-12-31", 5.0, "EUR")]
    assert raw["trailingPeRatio"] == [("2026-10-29", 20.0, None)]
    assert raw["annualEBITDA"] == []
    assert "/timeseries/A%2FB" in str(seen[0]) and "annualNetIncome" in str(seen[0])
    assert dict(seen[0].params)["period2"] == "1800000000"


def test_fetch_failure_raises_unavailable(monkeypatch):
    monkeypatch.setattr("app.yahoo.time.sleep", lambda s: None)
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(429)))
    with pytest.raises(DataUnavailableError):
        f.fetch_fundamentals("X", client=client)
