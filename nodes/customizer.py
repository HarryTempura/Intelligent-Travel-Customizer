from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph

from entities.customizer_state import CustomizerState
from nodes.questioner import questioner_node


def get_graph():
    """

    :return:
    """
    graph_builder = StateGraph(CustomizerState)

    graph_builder.add_node('questioner', questioner_node)

    graph_builder.set_entry_point('questioner')

    memory = InMemorySaver()

    return graph_builder.compile(checkpointer=memory)


def customizer_node(state=None):
    """

    :param state:
    :return:
    """
