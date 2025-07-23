from pydantic import BaseModel


class CustomizerDTO(BaseModel):
    messages: list = None
