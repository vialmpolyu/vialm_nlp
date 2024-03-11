from typing import Any, Dict, List

from NLP.vialm_llm import VialmLLM


class QRCodeAnalyzer():
    def __init__(
        self,
        model: str = "meta-llama/Llama-2-7b-chat-hf" 
    ) -> None:
        self._vialm_llm = VialmLLM(model) 
        with open('NLP/prompt/qrcode_analyzer_prompt.txt', 'r') as f:
            self._base_prompt = f"{''.join(f.readlines())}\n\n"

    def run_analysis(
            self, 
            data: List[Dict[str, Any]]
    ) -> List[str]:
        prompt = self._vialm_llm.create_prompt(data, self._base_prompt)
        response = self._vialm_llm.run_inference(prompt)

        return response
