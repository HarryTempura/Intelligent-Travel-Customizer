from pydantic import BaseModel, Field


class Analysis(BaseModel):
    demand: str = Field(description='本次旅行的基本信息和诉求')
    opt_recs: str = Field(description='需要继续确认的信息')
