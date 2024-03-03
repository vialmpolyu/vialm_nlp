from typing import Any, Dict, List

from NLP.vialm_llm import VialmLLM


class QRCodeAnalyzer():
    def __init__(
        self,
        data: List[Dict[str, Any]],
        prompt: str,
    ) -> None:
        self._data = data
        self._prompt = prompt

    def analyze(
            self, 
            model: str = "NLP/models/llama-2-7b"
    ) -> List[str]:
        
        vialm_llm = VialmLLM(model) 

        result = []

        for website in self._data:
            url = website['url']
            url_prompt = f"URL = {url}\n\n"
            prompt = self._prompt + url_prompt
            response = vialm_llm.run_llm(prompt)
            print(f"RESPONSE: {response}\n")
            print(f"LABEL: {website['description']}\n")
            result.append(response)

        return result
