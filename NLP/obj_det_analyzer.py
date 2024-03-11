from typing import Any, Dict, List, Union

from NLP.vialm_llm import VialmLLM


class ObjDetAnalyzer():
    def __init__(
        self,
        model: str = "meta-llama/Llama-2-7b-chat-hf" 
    ) -> None:
        self._llm = VialmLLM(model) 
        with open('NLP/prompt/obj_det_analyzer_prompt.txt', 'r') as f:
            self._base_prompt = f"{''.join(f.readlines())}\n\n"

    def run_analysis(
            self, 
            data: List[Dict[str, Any]]
    ) -> List[str]:
        prompt = self._llm.create_prompt(data, self._base_prompt)
        response = self._llm.run_inference(prompt)

        return response
    