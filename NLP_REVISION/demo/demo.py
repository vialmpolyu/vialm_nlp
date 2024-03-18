import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
parent = os.path.dirname(SCRIPT_DIR)
parent = os.path.dirname(parent)
sys.path.append(parent)

from NLP_REVISION.API_NLP import NLP_AI_AGENT

agent = NLP_AI_AGENT()
agent.init_module_nlp(CFG={})

task = agent.task_classify("Where is the qrcode?")
print(task)

res = agent.agent_ocr_qa("What is this url for: https://www.youtube.com/")
print(res)
