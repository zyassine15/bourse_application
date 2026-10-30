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


class Validation(BaseModel):
    status: str
    message: str


class AnalysisResponse(BaseModel):
    ticker: str
    as_of: str
    last_price: float
    change_1d: float | None
    model_version: str
    validation: Validation
    global_: GlobalResult = Field(alias="global", serialization_alias="global")
    horizons: dict[str, HorizonResult]
    indicators: dict[str, float]
    history: History
    disclaimer: str

    model_config = {"populate_by_name": True}
