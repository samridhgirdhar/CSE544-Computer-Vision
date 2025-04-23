#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering

# ------------ config ------------
IMAGE_PATH = "/home/iiitd/finetuningDeepseek/cva3/q1/sample_image.jpg"
device     = "cuda:1" if torch.cuda.is_available() else "cpu"

# ------------ load --------------
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model     = (
    BlipForQuestionAnswering
    .from_pretrained("Salesforce/blip-vqa-base")
    .to(device)
    .eval()
)

# ------------ helper ------------
def answer(q: str) -> str:
    image   = Image.open(IMAGE_PATH).convert("RGB")
    inputs  = processor(image, q, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_length=20)
    return processor.decode(out[0], skip_special_tokens=True)

# ------------ run ---------------
print("Dog:", answer("Where is the dog present in the image?"))
print("Man:", answer("Where is the man present in the image?"))


# Comment on accuracy:
# Both of those are spot on:
# 
# - **Dog: “in man’s arms”**  
#   The dog really is being cradled in his arms—no finer localization needed.  
# 
# - **Man: “living room”**  
#   The man is standing indoors in what looks like a living room (bookshelf, games, wood floor)—so “living room” is a perfectly reasonable answer to “Where is the man present?”  
# 
# In short, BLIP’s VQA head gave concise, accurate spatial answers for both questions.
