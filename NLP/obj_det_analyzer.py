from typing import Any, Dict, List, Union

from NLP.vialm_llm import VialmLLM


class ObjDetAnalyzer():
    def __init__(
        self,
        data: List[Dict[str, Any]],
        prompt: str,
    ) -> None:
        self._data = data
        self._prompt = prompt

    def evaluate(
            self, 
            model: str = "meta-llama/Llama-2-7b-chat-hf"
        ) -> List[Union[int, float]]:
            
            vialm_llm = VialmLLM(model) 

            score = 0
            mq_count = 0

            for img in self._data:
                item = img['cls']
                pos = img['boxxywh']
                item_prompt = f"ITEM=[{', '.join(item)}]\n"
                pos_prompt = f"POS=[{', '.join([str(box) for box in pos])}]\n"

                for mq in img['question list']:
                    question_prompt = f"QUESTION: {mq['question']}\n"
                    options_prompt = f"A: {mq['options']['A']}\nB: {mq['options']['B']}\nC: {mq['options']['C']} \nD: {mq['options']['D']}\n"

                    prompt = f"{self._prompt}\n{item_prompt}\n{pos_prompt}\n{question_prompt}\n{options_prompt}\nRESPONSE="
                    response = vialm_llm.run_inference(prompt)
                    print(f"RESPONSE: {response}\n")
                    print(f"ANSWER: {mq['answer']}\n")

                    mq_count += 1
                    if (response == mq['answer']):
                         score += 1

            return [score, mq_count, round(score/mq_count, 2)]
    