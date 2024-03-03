from typing import Any, Dict, List, Union

from NLP.vialm_llm import VialmLLM


class SceneAnalyzer():
    def __init__(
        self,
        data: List[Dict[str, Any]],
        prompt: str,
    ) -> None:
        self._data = data
        self._prompt = prompt

    def evaluate(
            self, 
            model: str = "NLP/models/llama-2-chat-7b-hf"
        ) -> List[Union[int, float]]:
            
            vialm_llm = VialmLLM(model) 

            score = 0
            mq_count = 0

            for img in self._data:
                item = img['cls']
                pos = img['boxxywh']
                item_prompt = f"ITEM=[{', '.join(item)}]\n\n"
                pos_prompt = f"POS=[{', '.join([str(box) for box in pos])}]\n\n"

                for mq in img['question list']:
                    question_prompt = f"QUESTION: {mq['question']}\n\n"
                    options_prompt = f"A: {mq['options']['A']}\nB: {mq['options']['B']}\nC: {mq['options']['C']} \nD: {mq['options']['D']}\n\n"

                    prompt = self._prompt + item_prompt + pos_prompt + question_prompt + options_prompt + "RESPONSE="
                    response = vialm_llm.run_llm(prompt)
                    print(f"RESPONSE: {response}, ANSWER: {mq['answer']}\n")

                    mq_count += 1
                    if (response == mq['answer']):
                         score += 1

            return [score, mq_count, round(score/mq_count, 2)]
    