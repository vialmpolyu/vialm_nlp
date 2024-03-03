import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
parent = os.path.dirname(SCRIPT_DIR)
parent = os.path.dirname(parent)
sys.path.append(parent)

import json

from NLP.qrcode_analyzer import QRCodeAnalyzer


data = json.load(open('NLP/data/qrcode_example.json'))
with open('NLP/utils/qrcode_analyzer_prompt.txt', 'r') as f:
    prompt = "".join(f.readlines()) + '\n\n'

analyzer = QRCodeAnalyzer(data, prompt)
result = analyzer.analyze()
