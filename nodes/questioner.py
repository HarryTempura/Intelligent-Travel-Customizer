from langchain_core.messages import SystemMessage, AIMessage

from common.llms import OLLAMA_QWEN3_4B
from entities.questions import Questions
from states.customizer_state import CustomizerState


def questioner_node(state: CustomizerState = None):
    """
    并收集用户的回答，以便后续制定个性化的旅行计划。

    :param state: 当前用户的状态或上下文，本例中未使用。
    :return: 返回一个字典，包含提问的问题列表和用户的回答。
    """
    print('=' * 40, 'questioner_node', '=' * 40)

    # 初始化大型语言模型，并指定输出结构为问题列表
    llm = OLLAMA_QWEN3_4B.with_structured_output(Questions)

    # 定义系统消息模板，包含旅行定制师的指令和注意事项
    sys_template = """
你是一名专业的旅行定制师。

### 指令

咨询客户并收集资料，明确出发时间出发时间、天数、预算范围，出发地和目的地，旅行方式偏好（自驾、跟车、包车、自助游等），行程节奏（松散休闲还是紧凑高效），同行人员（是否带孩子、老人、情侣、朋友）和偏好与忌讳（如不想走太多路，不喜欢博物馆、素食者等）等信息。输出准备提问客户的问题列表。

### 注意！避免任何额外的输出！这是旅行定制最关键的一步！{}
    """.strip()
    # 创建系统消息对象
    sys_message = SystemMessage(content=sys_template)
    # 定义 AI 消息模板
    ai_template = f"""
这是当前已经知晓的需求：{state.demand}
这是需继续明确和澄清的信息：{state.opt_recs}
    """.strip()
    # 创建 AI 消息对象
    ai_message = AIMessage(content=ai_template)

    # 调用大型语言模型生成问题列表
    response = llm.invoke([sys_message, ai_message])
    questions = response.questions

    answers = []
    # 遍历问题列表，与用户进行交互并收集回答
    for question in questions:
        print('Customizer:\n', question)

        answer = input('User:\n')
        answers.append(answer)

    # 返回问题和回答的字典
    return {'questions': questions, 'answers': answers}
