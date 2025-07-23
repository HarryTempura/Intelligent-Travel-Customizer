from dotenv import load_dotenv
from loguru import logger

from nodes.customizer import customizer_node

load_dotenv()


def start():
    customizer_node()


if __name__ == '__main__':
    logger.info('Start... ...')

    start()

    logger.info('End')
