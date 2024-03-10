from typing import Any, Dict, List
from enum import Enum

import torch
import transformers
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM


class TaskMode(Enum):
    OBJ_DET = "obj_det"
    OCR = "ocr"
    QRCODE = "qrcode"

    def __str__(self):
        return self.value
    

class VialmLLM():
    def __init__(
        self,
        model: str = "meta-llama/Llama-2-7b-chat-hf"
    ) -> None:
        if model == "THUDM/chatglm3-6b":
            self._tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
            self._model = AutoModel.from_pretrained(model, trust_remote_code=True).half().cuda()
        else:
            self._tokenizer = AutoTokenizer.from_pretrained(model)
            self._model = AutoModelForCausalLM.from_pretrained(model)

        self._pipeline = transformers.pipeline(
            "text-generation",
            model=self._model,
            tokenizer=self._tokenizer,
            torch_dtype=torch.float16,
            device_map="auto",
        )

    def create_prompt(
            self,
            data: List[Dict[str, Any]],
            base_prompt: str
    ) -> str:
        text = [item['text'] for item in data]
        position = [item['position'] for item in data]

        text_prompt = f"TEXT: [{', '.join(text)}]\n"
        position_prompt = f"POSITION: [{', '.join([str(coord) for coord in position])}]\n"

        prompt = f"{base_prompt}\n{text_prompt}\n{ position_prompt}\nANSWER: "
        return prompt

    def run_inference(
        self,
        input: str
    ) -> str:
        output = self._pipeline(
            input,
            do_sample=True,
            temperature=0.7,
            top_k=40,
            top_p=0.1,
            num_return_sequences=1,
            eos_token_id=self._tokenizer.eos_token_id,
            max_length=2048,
        )

        return output
    