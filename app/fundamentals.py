"""Données fondamentales (états financiers) des actions internationales, via l'endpoint « timeseries » de Yahoo.

Cet endpoint ne demande pas de clé et renvoie, pour chaque grandeur, les valeurs annuelles et « 12 derniers mois »
(TTM) publiées. Les ratios sont recalculés ici à partir de ces grandeurs, avec des règles explicites :

- une grandeur n'est utilisée que pour une **même période** (ex. résultat net et capitaux propres du même exercice) ;
- un ratio dénué de sens (résultat négatif, EBITDA négatif…) est marqué « non significatif » plutôt que calculé ;
- une donnée absente est marquée « indisponible » : on n'invente jamais de valeur.
"""

import time
from datetime import date
from urllib.parse import quote

import httpx

from app.yahoo import get_client, get_json

TIMESERIES_PATH = "/ws/fundamentals-timeseries/v1/finance/timeseries/{symbol}"
TYPES = (
    "trailingPeRatio",
    "trailingPegRatio",
    "trailingFreeCashFlow",
    "annualFreeCashFlow",
    "trailingTotalRevenue",
    "annualTotalRevenue",
    "trailingOperatingIncome",
    "annualOperatingIncome",
    "trailingNetIncome",
    "annualNetIncome",
    "annualStockholdersEquity",
    "annualInvestedCapital",
    "annualTaxRateForCalcs",
    "annualTotalDebt",
    "annualCashCashEquivalentsAndShortTermInvestments",
    "annualNetDebt",
    "annualEBITDA",
    "annualCashDividendsPaid",
)
HISTORY_YEARS = 6
# Un ratio de valorisation publié il y a plus longtemps que cela n'est plus celui d'aujourd'hui.
FRESH_DAYS = 60
DEFAULT_TAX_RATE = 0.25

# (date ISO, valeur, devise de publication)
Point = tuple[str, float, str | None]
Raw = dict[str, list[Point]]

OK, NOT_MEANINGFUL, UNAVAILABLE, NOT_APPLICABLE = "ok", "not_meaningful", "unavailable", "not_applicable"


def fetch_fundamentals(symbol: str, client: httpx.Client | None = None, now: float | None = None) -> Raw:
    now = int(now or time.time())
    payload = get_json(
        client or get_client(),
        TIMESERIES_PATH.format(symbol=quote(symbol, safe="")),
        {
            "type": ",".join(TYPES),
            "period1": now - HISTORY_YEARS * 365 * 86400,
            "period2": now,
            "merge": "false",
        },
        symbol,
    )
    raw: Raw = {}
    for item in payload.get("timeseries", {}).get("result") or []:
        key = item["meta"]["type"][0]
        points = []
        for p in item.get(key) or []:
            value = ((p or {}).get("reportedValue") or {}).get("raw")
            if value is not None and p.get("asOfDate"):
                points.append((p["asOfDate"], float(value), p.get("currencyCode")))
        raw[key] = sorted(points)
    return raw


def value(label, number=None, unit="number", status=OK, note=None, period=None, currency=None) -> dict:
    return {
        "label": label,
        "value": None if number is None else round(float(number), 4),
        "unit": unit,
        "currency": currency,
        "status": status,
        "note": note,
        "period": period,
    }


def item(item_id, label, help_text, values, reading=None) -> dict:
    return {"id": item_id, "label": label, "help": help_text, "reading": reading, "values": values}


def _series(raw: Raw, key: str) -> dict[str, Point]:
    return {p[0]: p for p in raw.get(key, [])}


def _latest_common(raw: Raw, *keys: str) -> str | None:
    dates = set.intersection(*(set(_series(raw, k)) for k in keys)) if keys else set()
    return max(dates) if dates else None


def _period(kind: str, as_of: str) -> str:
    return f"12 derniers mois au {as_of}" if kind == "ttm" else f"Exercice clos le {as_of}"


def _is_fresh(as_of: str, today: date) -> bool:
    return (today - date.fromisoformat(as_of)).days <= FRESH_DAYS


def _loss_note(raw: Raw) -> str | None:
    """Explique un résultat net négatif (12 derniers mois en priorité, sinon dernier exercice)."""
    for key in ("trailingNetIncome", "annualNetIncome"):
        pts = raw.get(key)
        if pts and pts[-1][1] <= 0:
            return "Résultat net négatif : ratio non significatif"
    return None


def per(raw: Raw, today: date) -> dict:
    help_text = "Cours / bénéfice par action (12 derniers mois) : le prix payé pour 1 de bénéfice annuel."
    pts = raw.get("trailingPeRatio")
    if pts and _is_fresh(pts[-1][0], today) and pts[-1][1] > 0:
        v = value("PER", pts[-1][1], "ratio", period=f"au {pts[-1][0]}")
    elif note := _loss_note(raw):
        v = value("PER", status=NOT_MEANINGFUL, note=note)
    else:
        v = value("PER", status=UNAVAILABLE, note="Non publié par la source")
    return item("per", "PER", help_text, [v])


def peg(raw: Raw, today: date) -> dict:
    help_text = "PER / croissance attendue des bénéfices : proche de 1, la croissance est payée à son juste prix."
    pts = raw.get("trailingPegRatio")
    if not pts or not _is_fresh(pts[-1][0], today):
        v = value("PEG", status=UNAVAILABLE, note="Non publié par la source")
    elif pts[-1][1] <= 0:
        v = value("PEG", status=NOT_MEANINGFUL, note="Croissance attendue négative ou nulle")
    else:
        v = value("PEG", pts[-1][1], "ratio", period=f"au {pts[-1][0]}", note="Basé sur les prévisions d'analystes")
    return item("peg", "PEG", help_text, [v])


def _pick_flow(raw: Raw, ttm_key: str, annual_key: str, *companions: tuple[str, str]) -> tuple[str, str] | None:
    """Choisit la période la plus récente (TTM ou annuelle) où la grandeur et ses compagnons existent."""
    best = None
    for kind, key, comps in (
        ("ttm", ttm_key, [c[0] for c in companions]),
        ("annual", annual_key, [c[1] for c in companions]),
    ):
        as_of = _latest_common(raw, key, *comps)
        if as_of and (best is None or as_of > best[1]):
            best = (kind, as_of)
    return best


def free_cash_flow(raw: Raw, currency: str | None) -> dict:
    help_text = "Cash généré après investissements (flux d'exploitation − capex) : ce qui reste pour dividendes, dette, rachats."
    label = "Free Cash Flow (FCF)"
    fcf_pick = _pick_flow(raw, "trailingFreeCashFlow", "annualFreeCashFlow")
    if not fcf_pick:
        return item("fcf", label, help_text, [value("FCF", status=UNAVAILABLE, note="Non publié par la source")])
    kind, as_of = fcf_pick
    fcf = _series(raw, "trailingFreeCashFlow" if kind == "ttm" else "annualFreeCashFlow")[as_of]
    revenue = _series(raw, "trailingTotalRevenue" if kind == "ttm" else "annualTotalRevenue").get(as_of)
    period = _period(kind, as_of)
    values = [value("FCF", fcf[1], "money", period=period, currency=fcf[2] or currency)]
    if revenue and revenue[1] > 0:
        values.append(value("Marge de FCF", fcf[1] / revenue[1], "pct", period=period))
    return item("fcf", label, help_text, values)


def roe_roic(raw: Raw) -> dict:
    help_text = "ROE = résultat net / capitaux propres. ROIC = résultat opérationnel après impôt / capital investi."
    # ROE
    as_of = _latest_common(raw, "annualNetIncome", "annualStockholdersEquity")
    if not as_of:
        roe = value("ROE", status=UNAVAILABLE, note="Non publié par la source")
    else:
        equity = _series(raw, "annualStockholdersEquity")[as_of][1]
        net_income = _series(raw, "annualNetIncome")[as_of][1]
        if equity <= 0:
            roe = value("ROE", status=NOT_MEANINGFUL, note="Capitaux propres négatifs ou nuls")
        else:
            roe = value("ROE", net_income / equity, "pct", period=_period("annual", as_of))
    # ROIC
    as_of = _latest_common(raw, "annualOperatingIncome", "annualInvestedCapital")
    if not as_of:
        roic = value("ROIC", status=UNAVAILABLE, note="Non publié par la source")
    else:
        capital = _series(raw, "annualInvestedCapital")[as_of][1]
        operating = _series(raw, "annualOperatingIncome")[as_of][1]
        tax_point = _series(raw, "annualTaxRateForCalcs").get(as_of)
        if tax_point and 0 <= tax_point[1] <= 0.5:
            tax, tax_note = tax_point[1], f"Impôt retenu : {tax_point[1]:.0%}"
        else:
            tax, tax_note = DEFAULT_TAX_RATE, f"Impôt supposé : {DEFAULT_TAX_RATE:.0%}"
        if capital <= 0:
            roic = value("ROIC", status=NOT_MEANINGFUL, note="Capital investi négatif ou nul")
        else:
            roic = value("ROIC", operating * (1 - tax) / capital, "pct", period=_period("annual", as_of), note=tax_note)
    return item("roe_roic", "ROE / ROIC", help_text, [roe, roic])


def _net_debt(raw: Raw, as_of: str) -> tuple[float, str] | None:
    """Dette nette de l'exercice et sa définition : la valeur publiée si elle existe, sinon dette totale − trésorerie."""
    reported = _series(raw, "annualNetDebt").get(as_of)
    if reported:
        return reported[1], "Dette nette publiée par la source"
    debt = _series(raw, "annualTotalDebt").get(as_of)
    cash = _series(raw, "annualCashCashEquivalentsAndShortTermInvestments").get(as_of)
    if debt and cash:
        return debt[1] - cash[1], "Dette totale − trésorerie et placements (baux inclus)"
    return None


def net_debt_to_ebitda(raw: Raw, currency: str | None) -> dict:
    help_text = "Dette nette / EBITDA : nombre d'années d'EBITDA pour rembourser la dette nette. Négatif = trésorerie nette."
    label = "Dette nette / EBITDA"
    ebitda = _series(raw, "annualEBITDA")
    as_of = next((d for d in sorted(ebitda, reverse=True) if _net_debt(raw, d) is not None), None)
    if as_of is None:
        unavailable = value(label, status=UNAVAILABLE, note="Non publié par la source")
        return item("net_debt_ebitda", label, help_text, [unavailable])
    net_debt, definition = _net_debt(raw, as_of)
    ebitda_value = ebitda[as_of][1]
    period = _period("annual", as_of)
    if ebitda_value <= 0:
        ratio = value(label, status=NOT_MEANINGFUL, note="EBITDA négatif ou nul")
    else:
        note = "Trésorerie nette (dette nette négative)" if net_debt < 0 else None
        ratio = value(label, net_debt / ebitda_value, "ratio", period=period, note=note)
    amount = value("Dette nette", net_debt, "money", period=period, currency=ebitda[as_of][2] or currency, note=definition)
    return item("net_debt_ebitda", label, help_text, [ratio, amount])


def operating_margin(raw: Raw) -> dict:
    help_text = (
        "Résultat opérationnel / chiffre d'affaires : la part du chiffre d'affaires qui reste après les coûts d'exploitation."
    )
    pick = _pick_flow(raw, "trailingOperatingIncome", "annualOperatingIncome", ("trailingTotalRevenue", "annualTotalRevenue"))
    label = "Marge opérationnelle"
    if not pick:
        return item("operating_margin", label, help_text, [value(label, status=UNAVAILABLE, note="Non publié par la source")])
    kind, as_of = pick
    income = _series(raw, "trailingOperatingIncome" if kind == "ttm" else "annualOperatingIncome")[as_of][1]
    revenue = _series(raw, "trailingTotalRevenue" if kind == "ttm" else "annualTotalRevenue")[as_of][1]
    if revenue <= 0:
        return item(
            "operating_margin", label, help_text, [value(label, status=UNAVAILABLE, note="Chiffre d'affaires indisponible")]
        )
    return item("operating_margin", label, help_text, [value(label, income / revenue, "pct", period=_period(kind, as_of))])


def payout_ratio(raw: Raw) -> dict:
    help_text = "Dividendes versés / résultat net : la part des bénéfices distribuée aux actionnaires."
    label = "Payout Ratio"
    income = raw.get("annualNetIncome")
    if not income:
        return item("payout", label, help_text, [value(label, status=UNAVAILABLE, note="Non publié par la source")])
    as_of, net_income, _ = income[-1]
    if net_income <= 0:
        return item(
            "payout", label, help_text, [value(label, status=NOT_MEANINGFUL, note=f"Résultat net négatif (exercice {as_of[:4]})")]
        )
    dividends = _series(raw, "annualCashDividendsPaid").get(as_of)
    if not dividends:
        return item("payout", label, help_text, [value(label, status=UNAVAILABLE, note="Dividendes non renseignés")])
    ratio = abs(dividends[1]) / net_income
    note = "Distribution supérieure au résultat net" if ratio > 1 else None
    return item("payout", label, help_text, [value(label, ratio, "pct", period=_period("annual", as_of), note=note)])


def compute_fundamentals(raw: Raw, today: date | None = None, currency: str | None = None) -> list[dict]:
    """Les sept indicateurs fondamentaux, dans l'ordre d'affichage."""
    today = today or date.today()
    return [
        per(raw, today),
        peg(raw, today),
        free_cash_flow(raw, currency),
        roe_roic(raw),
        net_debt_to_ebitda(raw, currency),
        operating_margin(raw),
        payout_ratio(raw),
    ]
