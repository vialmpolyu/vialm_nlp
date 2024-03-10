from typing import Any, Dict, List

from NLP.vialm_llm import VialmLLM


class QRCodeAnalyzer():
    def __init__(
        self,
        data: List[Dict[str, Any]],
        base_prompt: str,
    ) -> None:
        self._data = data
        with open('NLP/prompt/qrcode_analyzer_prompt.txt', 'r') as f:
            self._base_prompt = f"{''.join(f.readlines())}\n\n"

    def run_analysis(
            self, 
            model: str = "meta-llama/Llama-2-7b-chat-hf"
    ) -> List[str]:
        vialm_llm = VialmLLM(model) 
        prompt = vialm_llm.create_prompt(self._data, self._base_prompt)
        response = vialm_llm.run_inference(prompt)

        return response
