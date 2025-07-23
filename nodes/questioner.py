from langchain_core.messages import SystemMessage

from common.llms import OLLAMA_QWEN3_06B
from entities.questions import Questions


def questioner_node(state=None):
    """

    :param state:
    :return:
    """
    llm = OLLAMA_QWEN3_06B.with_structured_output(Questions)

    sys_template = """
你是一名专业的旅行定制师。

### 指令

咨询客户并收集资料，明确出发时间出发时间、天数、预算范围，出发地和目的地，旅行方式偏好（自驾、跟车、包车、自助游等），行程节奏（松散休闲还是紧凑高效），同行人员（是否带孩子、老人、情侣、朋友）和偏好与忌讳（如不想走太多路，不喜欢博物馆、素食者等）等信息。输出准备提问客户的问题列表。

### 注意！避免任何额外的输出！这是旅行定制最关键的一步！{}
    """.strip()
    sys_message = SystemMessage(content=sys_template)

    response = llm.invoke([sys_message])
    questions = response.questions

    return {'questions': questions}
