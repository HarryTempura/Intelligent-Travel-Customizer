from langchain_ollama import ChatOllama

from common.configs import configs

OLLAMA_GEMMA3_4B = ChatOllama(
    model=configs['ollama']['gemma3_4b'],
    temperature=configs['ollama']['temperature'],
    num_predict=configs['ollama']['num_predict']
)
OLLAMA_QWEN3_4B = ChatOllama(
    model=configs['ollama']['qwen3_4b'],
    temperature=configs['ollama']['temperature'],
    num_predict=configs['ollama']['num_predict']
)
OLLAMA_QWEN3_06B = ChatOllama(
    model=configs['ollama']['qwen3_06b'],
    temperature=configs['ollama']['temperature'],
    num_predict=configs['ollama']['num_predict']
)
