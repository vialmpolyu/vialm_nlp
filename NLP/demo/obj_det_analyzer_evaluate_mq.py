import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
parent = os.path.dirname(SCRIPT_DIR)
parent = os.path.dirname(parent)
sys.path.append(parent)

import json

from NLP.obj_det_analyzer import ObjDetAnalyzer


data = json.load(open('NLP/data/obj_det_example.json'))
with open('NLP/prompt/scene_analyzer_prompt.txt', 'r') as f:
    prompt = f"{''.join(f.readlines())}\n\n"

analyzer = ObjDetAnalyzer(data, prompt)
result = analyzer.evaluate()

# print(f"CORRECT: {result[0]}, TOTAL: {result[1]}, TEST SCORE: {result[2]}")
