# Copyright (c) 2023, Yulu Gan
# Licensed under the BSD 3-clause license (see LICENSE.txt)
# ** Description ** Rephrase the prompts
# --------------------------------------------------------
# References:
# https://huggingface.co/humarin/chatgpt_paraphraser_on_T5_base
# --------------------------------------------------------

# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# import random

# device = "cuda"
# tokenizer = AutoTokenizer.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base")
# model_re = AutoModelForSeq2SeqLM.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base")


# # def paraphrase(
# #     question,
# #     num_beams=5,
# #     num_beam_groups=5,
# #     num_return_sequences=5,
# #     repetition_penalty=10.0,
# #     diversity_penalty=3.0,
# #     no_repeat_ngram_size=2,
# #     temperature=0.7,
# #     max_length=128
# # ):
# #     input_ids = tokenizer(
# #         f'paraphrase: {question}',
# #         return_tensors="pt", padding="longest",
# #         max_length=max_length,
# #         truncation=True,
# #     ).input_ids
    
# #     outputs = model_re.generate(
# #         input_ids, temperature=temperature, repetition_penalty=repetition_penalty,
# #         num_return_sequences=num_return_sequences, no_repeat_ngram_size=no_repeat_ngram_size,
# #         num_beams=num_beams, num_beam_groups=num_beam_groups,
# #         max_length=max_length, diversity_penalty=diversity_penalty
# #     )

# #     res = tokenizer.batch_decode(outputs, skip_special_tokens=True)
# #     res.append(question)
# #     res.append(question.replace("help me",""))
        
# #     res = random.choice(res)

# #     return res

# class Rephrase(object):
    
#     def __init__(self, text, num_beams=20, 
#                  num_beam_groups=20, num_return_sequences=20,
#                  repetition_penalty=10.0, diversity_penalty=10.0,
#                  no_repeat_ngram_size=5, temperature=0.7, max_length=128
#                  ):
        
#         self.text = text
#         self.tokenizer              = AutoTokenizer.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base")
#         self.model                  = AutoModelForSeq2SeqLM.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base")
#         self.num_beams              = num_beams
#         self.num_beam_groups        = num_beam_groups
#         self.num_return_sequences   = num_return_sequences
#         self.repetition_penalty     = repetition_penalty
#         self.diversity_penalty      = diversity_penalty
#         self.no_repeat_ngram_size   = no_repeat_ngram_size
#         self.temperature            = temperature
#         self.max_length             = max_length
        
        
#     def do(self):
#         input_ids = self.tokenizer(
#             f'paraphrase: {self.text}',
#             return_tensors = "pt", padding="longest",
#             max_length = self.max_length,
#             truncation = True,
#         ).input_ids
        
#         outputs = self.model.generate(
#             input_ids, temperature = self.temperature, repetition_penalty = self.repetition_penalty,
#             num_return_sequences = self.num_return_sequences, no_repeat_ngram_size = self.no_repeat_ngram_size,
#             num_beams = self.num_beams, num_beam_groups = self.num_beam_groups,
#             max_length = self.max_length, diversity_penalty = self.diversity_penalty
#         )

#         res = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
#         # res.append(self.text)
#         # res.append(self.text.replace("help me",""))
        
#         # res = random.choice(res)

#         return res

# if __name__ == "__main__":
    
#     text = "Create a monocular depth map"
#     output = Rephrase(text).do()
#     # output.replace("percentage", "%")
#     print("output", type(output))
#     print("output", len(output))
#     print("output", output)


from revChatGPT.V1 import Chatbot
import pdb

response_all = ""

for line in open("./data/cls_prompts.txt"):
    line = line.strip()
    
    chatbot = Chatbot(config={
    "access_token": "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6Ik1UaEVOVUpHTkVNMVFURTRNMEZCTWpkQ05UZzVNRFUxUlRVd1FVSkRNRU13UmtGRVFrRXpSZyJ9.eyJodHRwczovL2FwaS5vcGVuYWkuY29tL3Byb2ZpbGUiOnsiZW1haWwiOiJnYW55dWx1QHN0dS5wa3UuZWR1LmNuIiwiZW1haWxfdmVyaWZpZWQiOnRydWV9LCJodHRwczovL2FwaS5vcGVuYWkuY29tL2F1dGgiOnsidXNlcl9pZCI6InVzZXItdWNXcFN6NXg4eXpEZXJRaWRkNDl6dXk3In0sImlzcyI6Imh0dHBzOi8vYXV0aDAub3BlbmFpLmNvbS8iLCJzdWIiOiJnb29nbGUtb2F1dGgyfDEwMTQ1NDA2MzgxNTg3MTI3MjMxNiIsImF1ZCI6WyJodHRwczovL2FwaS5vcGVuYWkuY29tL3YxIiwiaHR0cHM6Ly9vcGVuYWkub3BlbmFpLmF1dGgwYXBwLmNvbS91c2VyaW5mbyJdLCJpYXQiOjE2ODI5MTc2ODksImV4cCI6MTY4NDEyNzI4OSwiYXpwIjoiVGRKSWNiZTE2V29USHROOTVueXl3aDVFNHlPbzZJdEciLCJzY29wZSI6Im9wZW5pZCBwcm9maWxlIGVtYWlsIG1vZGVsLnJlYWQgbW9kZWwucmVxdWVzdCBvcmdhbml6YXRpb24ucmVhZCBvZmZsaW5lX2FjY2VzcyJ9.syCHJF6RuaDJfW4NIAfxdV10FpFHau6QHqNmu5NFqp5-tSu3EgLqjD2Stn851ZvguHOsUlKcTTnfR_dlUcJ9OPBKRsPehYTmxDzhEO0pLK5oKwOg8DdEOXuw9TYtf3GHIrneYW4UHplvXImHdyWhMEP_xo60cG58Y36RpmyH9R-dEELUST2A-9j83Ou11E_V9hpOSHyPJmQjdaGfm-HylEzd45G7XYnHgUX300PLlnwImLY_IPAkbtX6T7sY5BwWlL7TpR1ahgGSL915ZoAFQEf3eKBlhLyAxH3nJjcsgLQpa-VUlavAjSIRFwwYgJ0rWJgIaVGBscmAUx5Aa2ZSog"
    })

    prompt = "paraphrase this sentence and send me 10 different resluts: {}".format(line)
    response = ""

    for data in chatbot.ask(prompt):
        response = data["message"]
    
    response_all = response + response_all

pdb.set_trace()

print(response_all)
