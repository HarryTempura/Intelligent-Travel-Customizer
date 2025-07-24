from dotenv import dotenv_values
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

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

QWEN3_TURBO = ChatOpenAI(
    model=configs['qwen']['turbo'],
    temperature=configs['qwen']['temperature'],
    max_retries=configs['qwen']['max_retries'],
    base_url=configs['qwen']['base_url'],
    api_key=dotenv_values('.env')['DASHSCOPE_API_KEY']
)
