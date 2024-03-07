import torch
import transformers

from transformers import AutoModelForCausalLM, AutoTokenizer


class VialmLLM():
    def __init__(
        self,
        model: str = "meta-llama/Llama-2-7b-chat-hf"
    ) -> None:
        self._model = AutoModelForCausalLM.from_pretrained(model)
        self._tokenizer = AutoTokenizer.from_pretrained(model)

        self._pipeline = transformers.pipeline(
            "text-generation",
            model=self._model,
            tokenizer=self._tokenizer,
            torch_dtype=torch.float16,
            device_map="auto",
        )

    def run_inference(
        self,
        prompt: str
    ) -> str:
        response = self._pipeline(
            prompt,
            do_sample=True,
            temperature=0.7,
            top_k=40,
            top_p=0.1,
            num_return_sequences=1,
            eos_token_id=self._tokenizer.eos_token_id,
            max_length=2048,
        )

        return response
    