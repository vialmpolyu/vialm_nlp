from typing import Any, Dict, List

import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
    

class AgentLlama():
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-2-7b-chat-hf"
    ) -> None:
        self._model_name = model_name
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForCausalLM.from_pretrained(model_name)

        self._pipeline = transformers.pipeline(
            "text-generation",
            model=self._model,
            tokenizer=self._tokenizer,
            torch_dtype=torch.float16,
            device_map="auto",
        )

    def agent_completions(
        self,
        inputs: str
    ) -> str:
        outputs = self._pipeline(
            inputs,
            do_sample=True,
            temperature=0.7,
            top_k=40,
            top_p=0.1,
            num_return_sequences=1,
            eos_token_id=self._tokenizer.eos_token_id,
            max_new_tokens=200,
        )
        return outputs[0]['generated_text']
    