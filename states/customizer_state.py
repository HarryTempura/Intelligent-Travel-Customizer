from pydantic import BaseModel


class CustomizerState(BaseModel):
    questions: list[str] = None
    answers: list[str] = None
    demand: str = None
    opt_recs: str = None
