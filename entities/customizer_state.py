from pydantic import BaseModel


class CustomizerState(BaseModel):
    questions: list[str]
    answers: list[str]
