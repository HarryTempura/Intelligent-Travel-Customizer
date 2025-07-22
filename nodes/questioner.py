from langchain_core.prompts import SystemMessagePromptTemplate, ChatPromptTemplate

from common.llms import OLLAMA_GEMMA3_4B


def get_chain():
    """

    :return:
    """
    sys_template = """
你是一名专业的旅行定制师。

### 指令

咨询客户并收集资料，明确出发时间出发时间、天数、预算范围，出发地和目的地，旅行方式偏好（自驾、跟车、包车、自助游等），行程节奏（松散休闲还是紧凑高效），同行人员（是否带孩子、老人、情侣、朋友）和偏好与忌讳（如不想走太多路，不喜欢博物馆、素食者等）等信息。输出准备提问客户的问题列表。

### 注意！避免任何额外的输出！这是旅行定制最关键的一步！
    """
    sys_message = SystemMessagePromptTemplate.from_template(sys_template)
    prompt = ChatPromptTemplate.from_messages([sys_message])

    llm = OLLAMA_GEMMA3_4B.with_structured_output(list[str])

    chain = prompt | llm

    return chain

def questioner_node():
    chain=get_chain()


