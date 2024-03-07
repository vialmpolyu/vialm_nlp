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
            model: str = "meta-llama/Llama-2-7b-chat-hf"
    ) -> List[str]:
        
        vialm_llm = VialmLLM(model) 

        result = []

        for website in self._data:
            url = website['url']
            url_prompt = f"URL={url}\n"
            prompt = f"{self._prompt}\n{url_prompt}\nRESPONSE="
            response = vialm_llm.run_inference(prompt)
            print(f"RESPONSE: {response}\n")
            print(f"ANSWER: {website['description']}\n")
            result.append(response)

        return result
