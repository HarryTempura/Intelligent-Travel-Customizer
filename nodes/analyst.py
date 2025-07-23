import sys

from langchain_core.messages import HumanMessage
from langchain_tavily import TavilySearch
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState
from pydantic import BaseModel

from common.llms import OLLAMA_QWEN3_06B
from entities.analysis import Analysis
from entities.customizer_state import CustomizerState


class AnalystState(AgentState):
    demand: str
    opt_recs: str


def get_agent():
    tools = []
    tool = TavilySearch(max_result=7)
    tools.append(tool)

    llm = OLLAMA_QWEN3_06B.bind_tools(tools)

    sys_prompt = """
你是一名专业的旅行定制师。

### 指令

根据对话信息分析获得本次旅行的基本信息，并挖掘用户的诉求。然后总结好基本信息和诉求返回到 demand 字段。
如果你认为还有信息需要确认完善请将需要确认的信息返回到 opt_recs 字段，如果没有则将 None 返回到 opt_recs 字段。

### 注意！避免任何额外的输出！避免使用 markdown 格式！
    """.strip()

    react_agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=sys_prompt,
        response_format=Analysis,
        state_schema=AnalystState
    )

    return react_agent


def analyst_node(state=None):
    agent = get_agent()

    questions = state.questions
    answers = state.answers

    q_a = {}
    for i in range(len(questions)):
        question = questions[i]
        answer = answers[i]

        q_a[question] = answer

    human_template = f'这是我之前的对话信息，字典的 keys 是问题，字典的 values 是我的回答。{q_a}'
    human_message = HumanMessage(content=human_template)

    response = agent.invoke({'messages': [human_message]})
    analysis: Analysis = response.get('structured_response')

    return {'demand': analysis.demand, 'opt_recs': analysis.opt_recs}
