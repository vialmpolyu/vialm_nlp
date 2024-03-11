import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
parent = os.path.dirname(SCRIPT_DIR)
parent = os.path.dirname(parent)
sys.path.append(parent)

from NLP.qrcode_analyzer import QRCodeAnalyzer


data = json.load(open('NLP/data/qrcode.json'))

analyzer = QRCodeAnalyzer()
result = analyzer.run_analysis(data)
