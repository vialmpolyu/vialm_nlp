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
            model: str = "models/llama-2-7b"
    ) -> List[str]:
        
        vialm_llm = VialmLLM(model) 

        result = []

        for receipt in self._data:
            content = receipt['content']
            content_prompt = f"CONTENT = {content}\n\n"
            prompt = self._prompt + content_prompt
            response = vialm_llm.run_llm(prompt)
            print(f"RESPONSE: {response}, ANSWER: {receipt['answer']}")
            result.append(response)

        return result
