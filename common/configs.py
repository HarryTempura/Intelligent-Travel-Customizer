import yaml

with open('./config.yaml', 'r', encoding='utf-8') as file:
    configs = yaml.safe_load(file)
