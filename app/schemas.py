from pydantic import BaseModel, Field


class HorizonResult(BaseModel):
    label: str
    days: int
    score: float = Field(ge=0, le=100)
    verdict: str
    rule_score: float
    ml_score: float | None
    components: dict[str, float]


class GlobalResult(BaseModel):
    score: float = Field(ge=0, le=100)
    verdict: str


class History(BaseModel):
    dates: list[str]
    series: dict[str, list[float | None]]


class PriceHistory(BaseModel):
    dates: list[str]
    close: list[float]
    sma50: list[float | None]
    sma200: list[float | None]


class Liquidity(BaseModel):
    traded_days_ratio: float = Field(ge=0, le=1)
    level: str
    warning: str | None


class Validation(BaseModel):
    status: str
    message: str


class AnalysisResponse(BaseModel):
    ticker: str
    name: str
    currency: str
    market: str
    kind: str
    leverage: int
    notice: str | None
    as_of: str
    last_price: float
    change_1d: float | None
    model_version: str
    validation: Validation
    liquidity: Liquidity
    global_: GlobalResult = Field(alias="global", serialization_alias="global")
    horizons: dict[str, HorizonResult]
    history: History
    prices: PriceHistory
    disclaimer: str

    model_config = {"populate_by_name": True}


class PricesResponse(BaseModel):
    """Vue « cours seul » : valeurs sans assez d'historique pour calculer un score."""

    ticker: str
    name: str
    currency: str
    market: str
    kind: str
    leverage: int
    notice: str | None
    as_of: str
    last_price: float
    change_1d: float | None
    prices: PriceHistory
    reason: str | None


class IndicatorValue(BaseModel):
    label: str
    value: float | None
    unit: str  # ratio, pct, money, price, number, points
    currency: str | None
    status: str  # ok, not_meaningful, unavailable, not_applicable
    note: str | None
    period: str | None


class IndicatorItem(BaseModel):
    id: str
    label: str
    help: str
    reading: str | None
    values: list[IndicatorValue]


class IndicatorGroup(BaseModel):
    id: str
    label: str
    status: str
    note: str | None
    items: list[IndicatorItem]


class IndicatorsResponse(BaseModel):
    ticker: str
    as_of: str
    currency: str
    groups: list[IndicatorGroup]
