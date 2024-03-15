import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
parent = os.path.dirname(SCRIPT_DIR)
parent = os.path.dirname(parent)
sys.path.append(parent)

from NLP.task_distributer import TaskDistributer


data = json.load(open('NLP/data/task.json'))

analyzer = TaskDistributer()
result = analyzer.run_analysis(data)
