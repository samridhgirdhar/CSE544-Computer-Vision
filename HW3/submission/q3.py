#!/usr/bin/env python
# coding: utf-8

# In[3]:


import warnings
warnings.filterwarnings('ignore')


# In[1]:


import os, glob, zipfile, json, math
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from PIL import Image
import torch

# HuggingFace
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    CLIPProcessor,
    CLIPModel,
)

DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
EXTRACT_DIR = "samples"


# In[5]:


Path(EXTRACT_DIR).mkdir(parents=True, exist_ok=True)

IMAGE_PATHS = sorted(glob.glob(os.path.join(EXTRACT_DIR, "*")))
print(f"Found {len(IMAGE_PATHS)} images")


# In[2]:


print("Loading BLIP captioning model …")
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model     = (
    BlipForConditionalGeneration
    .from_pretrained("Salesforce/blip-image-captioning-base")
    .to(DEVICE)
    .eval()
)


# In[8]:


def generate_caption(img: Image.Image) -> str:
    inputs = blip_processor(img, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = blip_model.generate(**inputs, max_length=30)
    return blip_processor.decode(out[0], skip_special_tokens=True)

captions = {}
for img_path in tqdm(IMAGE_PATHS, desc="Captioning"):
    img = Image.open(img_path).convert("RGB")
    captions[img_path] = generate_caption(img)

print(json.dumps(list(captions.items())[:11], indent=2))  # preview first 3


# In[7]:


clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE).eval()


# In[9]:


def clip_similarity(img: Image.Image, text: str) -> float:
    inputs = clip_processor(text=[text], images=img, return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        outs = clip_model(**inputs)
    img_emb  = outs.image_embeds / outs.image_embeds.norm(dim=-1, keepdim=True)
    txt_emb  = outs.text_embeds  / outs.text_embeds.norm(dim=-1, keepdim=True)
    return float((img_emb * txt_emb).sum())

def clip_score(sim: float) -> float:
    """
    CLIPScore / CLIPS = cosine similarity * 100
    (Hessel et al. 2021). No length penalty for simplicity.
    """
    return sim * 100.0


# In[12]:


rows = []
for img_path, caption in tqdm(captions.items(), desc="Evaluating with CLIP"):
    image = Image.open(img_path).convert("RGB")
    cos   = clip_similarity(image, caption)
    score = clip_score(cos)
    rows.append(
        dict(
            image=os.path.basename(img_path),
            caption=caption,
            clip_cosine=cos,
            clip_score=score,
        )
    )

df = pd.DataFrame(rows)
df.to_csv("blip_clip_results.csv", index=False)
df.head(10)


# 
# 
# | Metric | What it measures | Good for | Caveats |
# |---|---|---|---|
# | **Cosine similarity (raw CLIP)** | Angular distance between CLIP image & text embeddings | Quick sanity‑check of semantic match; ranking captions for one image | Uncalibrated; values vary with model / layer; not directly comparable across setups |
# | **CLIPScore / CLIPS** | Cosine × 100 (sometimes length‑penalized) | Reporting caption quality with a single number; correlates well with human judgment | Still inherits CLIP bias; higher isn’t always better for specificity vs. generality |
# | **CIDEr** | n‑gram TF‑IDF similarity against multiple references | Traditional caption benchmarks (COCO, Flickr); rewards consensus wording | Needs ground‑truth reference captions—unavailable for web images or zero‑shot tasks |
# | **SPICE** | Scene‑graph overlap (objects, attributes, relations) | Evaluating semantic correctness beyond surface wording | Slower; depends on reliable scene‑graph parsing; again needs reference captions |
# | **BLEU / ROUGE / METEOR** | n‑gram overlap | Historical baselines; cheap to compute | Weak correlation with human judgment, especially for open‑vocabulary captions |
# | **Image–Text Retrieval Recall (R@k)** | Does the caption retrieve its own image among distractors? | Dataset‑level evaluation of alignment models | Requires a large gallery; only yields set‑level statistics, not per‑caption scores |
# | **Human evaluation** | Direct judgment of relevance, fluency, detail | Final QA, user‑facing applications | Expensive and slow; subjective variance |
# 
# **When to use what**
# 
# - **Exploratory or zero‑shot settings** (no reference captions): CLIP cosine / CLIPScore are handy—immediate, reference‑free, and correlate reasonably with human assessments.
# - **Model development on COCO‑style datasets**: pair CLIPScore with CIDEr or SPICE, so you capture both semantic alignment and lexical diversity.
# - **Application‑specific tuning** (e.g., product search captions): perform **retrieval recall**—does the caption uniquely find its image among similar items?
# - **Deployment‑critical outputs** (medical, legal): always add a round of **human evaluation**, even if automated scores look high.
# 
