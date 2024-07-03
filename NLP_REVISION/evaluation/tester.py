import json
import os
from openai import OpenAI

f = open('NLP_REVISION/evaluation/test_data.json')
test_data = json.load(f)

client = OpenAI(
    # This is the default and can be omitted
    # e09d8e00f0371c3366b2e023a277798fd32e903d1ddb70323b76fa623cca5682
    # bb80e96d38c4e1e43120b0e370d499778d6af8370a498c402537ecda9aad83d9
    api_key=os.environ.get("OPENAI_API_KEY"),
)

res = []

for data in test_data:
    # response = client.chat.completions.create(
    #     messages=[
    #         {
    #             "role": "user",
    #             "content": f"{data['instruction']}\nf{data['input']}",
    #         }
    #     ],
    #     model="gpt-3.5-turbo",
    # )
    # print(response)
    res.append(f"{data['instruction']}\nf{data['input']}")

with open('NLP_REVISION/evaluation/input.json', 'w', encoding ='utf8') as json_file: 
    json.dump(res, json_file, indent=4) 
