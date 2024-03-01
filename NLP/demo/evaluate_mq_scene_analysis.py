import json

from scene_analysis import SceneAnalysis

f = open('../data/scene_analysis_testset.json')
data = json.load(f)

tester = SceneAnalysis(data)
result = tester.evaluate_score()

print(f"CORRECT: {result[0]}, TOTAL: {result[1]}, TEST SCORE: {result[2]}")
