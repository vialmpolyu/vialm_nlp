import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
parent = os.path.dirname(SCRIPT_DIR)
parent = os.path.dirname(parent)
sys.path.append(parent)

import json

from NLP.scene_analysis import SceneAnalysis


data = json.load(open('NLP/data/scene_analysis_testset.json'))
with open('NLP/utils/scene_analysis_prompt.txt', 'r') as f:
    prompt = "".join(f.readlines()) + '\n\n'

tester = SceneAnalysis(data, prompt)
result = tester.evaluate_score()

print(f"CORRECT: {result[0]}, TOTAL: {result[1]}, TEST SCORE: {result[2]}")
