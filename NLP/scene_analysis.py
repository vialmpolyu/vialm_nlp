from typing import Any, Dict, List, Union

from NLP.vialm_llm import VialmLLM


class SceneAnalysis():
    def __init__(
        self,
        data: List[Dict[str, Any]]
    ) -> None:
        self._data = data

    def evaluate_score(
            self, 
            model: str = "models/llama-2-7b"
        ) -> List[Union[int, float]]:
            
            vialm_llm = VialmLLM(model) 

            score = 0
            mq_count = 0
            with open('NLP/utils/scene_analysis_prompt.txt', 'r') as f:
                prompt = "".join(f.readlines()) + '\n\n'

            for img in self._data:
                item = img['cls']
                pos = img['boxxywh']
                item_prompt = "ITEM = [" + ", ".join(item) + "]\n\n"
                pos_prompt = "POS = [" + ", ".join([str(box) for box in pos]) + "]\n\n"

                for mq in img['question list']:
                    question_prompt = "QUESTION: " + mq['question'] + "\n\n"
                    options_prompt = "A: " + mq['options']['A'] + "\nB: " + mq['options']['B'] + "\nC: " + mq['options']['C'] + "\nD: " + mq['options']['D'] + "\n\n"

                    prompt = prompt + item_prompt + pos_prompt + question_prompt + options_prompt
                    response = vialm_llm.run_llm(prompt)
                    print(f"RESPONSE: {response}, ANSWER: {mq['answer']}")

                    mq_count += 1
                    if (response == mq['answer']):
                         score += 1

            return [score, mq_count, round(score/mq_count, 2)]
    