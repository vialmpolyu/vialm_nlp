import numpy as np
import json

from NLP_REVISION.nlpmodules.chat_llama import AgentLlama
# from NLP.nlpmodules.policy_speech import change_lang, change_sex, change_speaker



class NLP_AI_AGENT:
    def __init__(self):
        self._ai_agent = None

    def init_module_nlp(self, CFG):
        self._lang_ind = 'zh_CN'
        model_name = 'meta-llama/Llama-2-7b-chat-hf'
        prompt_fn = 'NLP_REVISION/prompt.json'

        openai_chat_module = AgentLlama(model_name)
        
        with open(prompt_fn, 'r', encoding='utf-8') as f:
            self._prompt_obj = json.load(f)

        self._ai_agent = openai_chat_module

    def task_classify(self, query_str):
        if query_str == None or len(query_str) < 5:
            return None
        base_prompt = self._prompt_obj['TASK_DISPATCHER'][0][self._lang_ind]
        query_str = f"{base_prompt}\n{query_str}\nANSWER: "
        res = self._ai_agent.agent_completions(query_str)
        return res
    
    def agent_obj_rpt(self, query_str):
        if query_str == None or len(query_str) < 5:
            return None
        base_prompt = self._prompt_obj['OBJ_REPORT'][0][self._lang_ind]
        query_str = f"{base_prompt}\n{query_str}\nANSWER: "
        res = self._ai_agent.agent_completions(query_str)
        return res
    
    def agent_obj_qa(self, query_str):
        if query_str == None or len(query_str) < 5:
            return None
        base_prompt = self._prompt_obj['OBJ_QA'][0][self._lang_ind]
        query_str = f"{base_prompt}\n{query_str}\nANSWER: "
        res = self._ai_agent.agent_completions(query_str)
        return res
        
    def agent_ocr_rpt(self, query_str):
        if query_str == None or len(query_str) < 5:
            return None
        base_prompt = self._prompt_obj['OCR_REPORT'][0][self._lang_ind]
        query_str = f"{base_prompt}\n{query_str}\nANSWER: "
        res = self._ai_agent.agent_completions(query_str)
        return res

    def agent_ocr_qa(self, query_str):
        if query_str == None or len(query_str) < 5:
            return None
        base_prompt = self._prompt_obj['OCR_QA'][0][self._lang_ind]
        query_str = f"{base_prompt}\n{query_str}\nANSWER: "
        res = self._ai_agent.agent_completions(query_str)
        return res
    
    def agent_qrcode_rpt(self, query_str):
        if query_str == None or len(query_str) < 5:
            return None
        base_prompt = self._prompt_obj['QRCODE_REPORT'][0][self._lang_ind]
        query_str = f"{base_prompt}\n{query_str}\nANSWER: "
        res = self._ai_agent.agent_completions(query_str)
        return res
    
    def agent_qrcode_qa(self, query_str):
        if query_str == None or len(query_str) < 5:
            return None
        base_prompt = self._prompt_obj['QRCODE_QA'][0][self._lang_ind]
        query_str = f"{base_prompt}\n{query_str}\nANSWER: "
        res = self._ai_agent.agent_completions(query_str)
        return res
    
    # def agent_setting(self, 
    #                   query_str, 
    #                   CFG, 
    #                   asrobj, 
    #                   ttsobj):
        
    #     if change_lang(query_str, CFG, asrobj, ttsobj):
    #         ttsobj.voice_change_speech_lang(CFG)
    #     elif change_sex(query_str, CFG, asrobj, ttsobj):
    #         ttsobj.voice_change_speech_sex(CFG)
    #     elif change_speaker(query_str, CFG, asrobj, ttsobj):
    #         ttsobj.voice_change_speech_speaker(CFG)
    #     else:
    #         if query_str == None:
    #             return
    #         else:
    #             res = self._ai_agent.agent_completions(query_str)
    #             # res = self.chatagent.azure_chat(query_str)

    #             ttsobj.voice_customize(res)
    #             # asyncio.run(tts.text_to_speech_and_play('嗯'+res))  # 如果用Edgetts需要使用异步执行

    # def agent_general(self, query_str):
    #     if query_str == None or len(query_str) < 5:
    #         return None
    #     query_str = f"{query_str}\nANSWER: "
    #     res = self._ai_agent.agent_completions(query_str)
    #     return res
