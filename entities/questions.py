from typing import List

from pydantic import BaseModel, Field


class Questions(BaseModel):
    questions: List[str] = Field(description='准备提问客户的问题列表')
