import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
parent = os.path.dirname(SCRIPT_DIR)
parent = os.path.dirname(parent)
sys.path.append(parent)

import json

from NLP.scene_analyzer import SceneAnalyzer


data = json.load(open('NLP/data/scene_testset.json'))
with open('NLP/utils/scene_analyzer_prompt.txt', 'r') as f:
    prompt = f"{''.join(f.readlines())}\n\n"

analyzer = SceneAnalyzer(data, prompt)
result = analyzer.evaluate()

# print(f"CORRECT: {result[0]}, TOTAL: {result[1]}, TEST SCORE: {result[2]}")
