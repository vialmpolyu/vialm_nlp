import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
parent = os.path.dirname(SCRIPT_DIR)
parent = os.path.dirname(parent)
sys.path.append(parent)

from NLP.ocr_analyzer import OCRAnalyzer


data = json.load(open('NLP/data/ocr.json'))

analyzer = OCRAnalyzer(data)
result = analyzer.run_analysis()
