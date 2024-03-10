import argparse
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
parent = os.path.dirname(SCRIPT_DIR)
parent = os.path.dirname(parent)
sys.path.append(parent)

from NLP.vialm_llm import TaskMode
from NLP.obj_det_analyzer import ObjDetAnalyzer
from NLP.ocr_analyzer import OCRAnalyzer
from NLP.qrcode_analyzer import QRCodeAnalyzer


def main(args):
    model_path = args.model_path
    task_mode = args.task_mode
    data_path = args.data_path

    data = json.load(open(data_path))

    if task_mode == TaskMode.OBJ_DET:
        analyzer = ObjDetAnalyzer(data)
    elif task_mode == TaskMode.OCR:
        analyzer = OCRAnalyzer(data)
    elif task_mode == TaskMode.QRCODE:
        analyzer = QRCodeAnalyzer(data)
    else:
        raise ValueError("Invalid TaskMode")
    
    result = analyzer.run_analysis(model_path)
    print(result)

""" sample run:
python NLP/script/run_vialm_llm.py  --model_path=meta-llama/Llama-2-7b-chat-hf --task_mode=qrcode --data_path=NLP/data/qrcode.json
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model.",
    )
    parser.add_argument(
        "--task_mode",
        type=TaskMode,
        required=True,
        help="Task mode for the model.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the data.",
    )   
    args = parser.parse_args()
    main(args)
