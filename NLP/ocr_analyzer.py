from typing import Any, Dict, List

from NLP.vialm_llm import VialmLLM


class OCRAnalyzer():
    def __init__(
        self,
        data: List[Dict[str, Any]],
        prompt: str,
    ) -> None:
        self._data = data
        self._prompt = prompt

    def analyze(
            self,
            model: str = "NLP/models/llama-2-chat-7b-hf"
    ) -> List[str]:
        
        vialm_llm = VialmLLM(model) 

        result = []

        for receipt in self._data:
            content = receipt['content']
            content_prompt = f"CONTENT={content}\n"
            prompt = f"{self._prompt}\n{content_prompt}\nRESPONSE="
            response = vialm_llm.run_llm(prompt)
            print(f"RESPONSE: {response}\n")
            print(f"ANSWER: {receipt['answer']}\n")
            result.append(response)

        return result
