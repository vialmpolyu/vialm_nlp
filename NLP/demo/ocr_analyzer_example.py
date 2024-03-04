import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
parent = os.path.dirname(SCRIPT_DIR)
parent = os.path.dirname(parent)
sys.path.append(parent)

import json

from NLP.ocr_analyzer import OCRAnalyzer


data = json.load(open('NLP/data/ocr_example.json'))
with open('NLP/utils/ocr_analyzer_prompt.txt', 'r') as f:
    prompt = f"{''.join(f.readlines())}\n\n"

analyzer = OCRAnalyzer(data, prompt)
result = analyzer.analyze()
