import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
parent = os.path.dirname(SCRIPT_DIR)
parent = os.path.dirname(parent)
sys.path.append(parent)

from NLP.obj_det_analyzer import ObjDetAnalyzer


data = json.load(open('NLP/data/obj_det.json'))

analyzer = ObjDetAnalyzer()
result = analyzer.run_analysis(data)
