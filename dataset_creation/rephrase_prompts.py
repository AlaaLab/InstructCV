from parrot import Parrot
import torch
import warnings
warnings.filterwarnings("ignore")

''' 
uncomment to get reproducable paraphrase generations
def random_state(seed):
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

random_state(1234)
'''

#Init models (make sure you init ONLY once if you integrate this to your code)
def rephrase(phrases):
    
    parrot = Parrot(model_tag="prithivida/parrot_paraphraser_on_T5")

    for phrase in phrases:
        print("-"*100)
        print("Input_phrase: ", phrase)
        print("-"*100)
        para_phrases = parrot.augment(input_phrase=phrase, use_gpu=False)
    for para_phrase in para_phrases:
        print(para_phrase)

if __name__ == "__main__":
    
    phrases = ["Can you help me detect the dog?",
           "What are the famous places we should not miss in Russia?"]
    
    rephrase(phrases)
