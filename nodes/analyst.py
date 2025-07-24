from langchain_core.messages import HumanMessage
from langchain_tavily import TavilySearch
from langgraph.prebuilt import create_react_agent

from common.llms import OLLAMA_QWEN3_4B
from entities.analysis import Analysis
from entities.customizer_dto import CustomizerDTO


def get_agent():
    """
    创建并返回一个智能体(agent)

    :return: 配置好的react_agent智能体实例，可用于处理旅行定制相关对话
    """
    # 初始化工具列表，用于扩展智能体功能
    tools = []
    # 添加 Tavily 搜索引擎工具，用于获取旅行相关信息
    tool = TavilySearch(max_result=7)
    tools.append(tool)

    # 绑定工具到大语言模型，创建具备工具调用能力的LLM实例
    llm = OLLAMA_QWEN3_4B

    # 定义系统提示词，明确智能体的角色和任务要求
    sys_prompt = """
你是一名专业的旅行定制师。

### 指令

根据对话信息分析获得本次旅行的基本信息，并挖掘用户的诉求。然后总结好基本信息和诉求返回到 demand 字段。
如果你认为还有信息需要确认完善请将需要确认的信息返回到 opt_recs 字段，如果没有则将 None 返回到 opt_recs 字段。

### 注意！避免任何额外的输出！避免使用 markdown 格式！
    """.strip()

    # 创建react_agent智能体，整合模型、工具、提示词和输出格式
    react_agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=sys_prompt,
        response_format=Analysis
    )

    return react_agent


def analyst_node(state=None):
    """
    分析节点函数，用于分析用户的问题和回答，提取需求和优化建议

    :param state: 状态对象，包含用户的问题列表和回答列表
    :return: 包含用户需求和优化建议的字典，格式为{'demand': 需求, 'opt_recs': 优化建议列表}
    """
    agent = get_agent()

    # 提取问题和回答列表
    questions = state.questions
    answers = state.answers

    # 将问题和回答组合成字典映射
    q_a = {}
    for i in range(len(questions)):
        question = questions[i]
        answer = answers[i]

        q_a[question] = answer

    # 构造人类消息模板，包含问题回答的对话信息
    human_template = f'这是我之前的对话信息，字典的 keys 是问题，字典的 values 是我的回答。{q_a}'
    human_message = HumanMessage(content=human_template)

    dto = CustomizerDTO(messages=[human_message])

    # 调用agent进行分析处理
    response = agent.invoke(dto)
    analysis: Analysis = response.get('structured_response')

    # 返回分析结果，包括需求和优化建议
    return {'demand': analysis.demand, 'opt_recs': analysis.opt_recs}


def analyst_route(state):
    """
    根据用户的状态分析并返回其路由方向

    :param state: 包含用户状态信息的字典，用于判断用户的路由方向
    :return: 如果状态中包含 'none'（不区分大小写），则返回空字符串，否则返回 'questioner'
    """
    # 从用户状态中提取'opt_recs'字段的值
    opt_recs = state.opt_recs

    # 判断'opt_recs'字段的值是否包含 'none'（不区分大小写）
    if 'none' in opt_recs.lower():
        # 如果包含 'none'，则返回空字符串，表示不进行路由
        return ''
    else:
        # 如果不包含 'none'，则返回 'questioner'，表示路由到提问者
        return 'questioner'
