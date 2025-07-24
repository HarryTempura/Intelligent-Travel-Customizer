from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph

from entities.customizer_dto import CustomizerDTO
from entities.customizer_state import CustomizerState
from nodes.analyst import analyst_node, analyst_route
from nodes.questioner import questioner_node


def get_graph():
    """

    :return:
    """
    graph_builder = StateGraph(CustomizerState)

    graph_builder.add_node('questioner', questioner_node)
    graph_builder.add_node('analyst', analyst_node)

    graph_builder.set_entry_point('questioner')

    graph_builder.add_edge('questioner', 'analyst')
    graph_builder.add_conditional_edges('analyst', analyst_route)

    memory = InMemorySaver()

    return graph_builder.compile(checkpointer=memory)


def customizer_node(state=None):
    """

    :param state:
    :return:
    """
    graph = get_graph()

    input_t = CustomizerDTO()
    config = RunnableConfig(configurable={'thread_id': 1})

    events = graph.stream(input_t, config, stream_mode='values')
    for chunk in events:
        print('=' * 80)
        print(chunk)
