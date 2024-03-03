import torch
import transformers

from transformers import LlamaForCausalLM, LlamaTokenizer


class VialmLLM():
    def __init__(
        self,
        model: str = "NLP/models/llama-2-chat-7b-hf"
    ) -> None:
        self._model = LlamaForCausalLM.from_pretrained(model)
        self._tokenizer = LlamaTokenizer.from_pretrained(model)

        self._pipeline = transformers.pipeline(
            "text-generation",
            model=self._model,
            tokenizer=self._tokenizer,
            torch_dtype=torch.float16,
            device_map="auto",
        )

    def run_llm(
            self,
            prompt: str
    ) -> str:
        response = self._pipeline(
            prompt,
            do_sample=True,
            top_k=40,
            num_return_sequences=1,
            eos_token_id=self._tokenizer.eos_token_id,
            max_length=1600,
        )

        return response
    