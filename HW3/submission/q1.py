#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

# Extract results
clip_scores = [5.95, 78.37, 0.14, 0.01, 1.21, 0.02, 0.02, 0.00, 0.63, 13.62]
clips_scores = [76.11, 16.04, 0.01, 0.00, 1.57, 0.00, 0.00, 0.03, 0.77, 5.46]
labels = [f"Caption {i+1}" for i in range(10)]

# Create plot
x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))
rects1 = ax.bar(x - width/2, clip_scores, width, label='CLIP')
rects2 = ax.bar(x + width/2, clips_scores, width, label='CLIPS')

# Add labels and legend
ax.set_xlabel('Captions')
ax.set_ylabel('Similarity Score (%)')
ax.set_title('Comparison of CLIP vs CLIPS Similarity Scores')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

plt.tight_layout()
plt.show()


# In[1]:


get_ipython().system('export CUDA_VISIBLE_DEVICES=1')


# In[4]:


get_ipython().system('conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0')
get_ipython().system('pip install ftfy regex tqdm')
get_ipython().system('pip install git+https://github.com/openai/CLIP.git')


# In[ ]:


# Install required dependencies
import torch
import clip
from PIL import Image
import requests
from io import BytesIO
import numpy as np


image = Image.open("sample_image.jpg")

# Load the CLIP model with pretrained weights (ViT-B/32)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model, preprocess = clip.load("ViT-B/32", device=device)

# Preprocess the image
image_input = preprocess(image).unsqueeze(0).to(device)

# Define 10 random textual descriptions
captions = [
    "A man holding a large dog",
    "A gray Great Dane being held by its owner",
    "A person with a small puppy",
    "A dog sitting on a couch",
    "A man in formal attire with a pet",
    "A woman holding a cat",
    "A gray horse in a stable",
    "A person standing next to a bookshelf",
    "A large dog with its owner in a living room",
    "A man in a white shirt holding a gray dog"
]

# Tokenize the text
text_inputs = torch.cat([clip.tokenize(c) for c in captions]).to(device)

# Calculate features and similarities
with torch.no_grad():
    image_features = model.encode_image(image_input)
    text_features = model.encode_text(text_inputs)
    
    # Normalize features
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    # Calculate similarity scores
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    
    # Print the results
    print("\nCLIP Similarity Scores:")
    for i, caption in enumerate(captions):
        print(f"{caption}: {similarity[0][i].item():.2%}")


# In[10]:


get_ipython().system('pip3 install -r requirements.txt')
get_ipython().system('pip install open_clip_torch')


# In[ ]:


# Install required dependencies for CLIPS
import torch
import torch.nn.functional as F
from PIL import Image
from open_clip import create_model_from_pretrained, get_tokenizer

# Load the CLIPS-Large-14-224 model from HuggingFace
model, preprocess = create_model_from_pretrained('hf-hub:UCSC-VLAA/ViT-L-14-CLIPS-Recap-DataComp-1B')
tokenizer = get_tokenizer('hf-hub:UCSC-VLAA/ViT-L-14-CLIPS-Recap-DataComp-1B')

# Load the same image used for CLIP
image = Image.open("sample_image.jpg")
image_input = preprocess(image).unsqueeze(0)

# Use the same 10 captions from the CLIP example
captions = [
    "A man holding a large dog",
    "A gray Great Dane being held by its owner",
    "A person with a small puppy",
    "A dog sitting on a couch",
    "A man in formal attire with a pet",
    "A woman holding a cat",
    "A gray horse in a stable",
    "A person standing next to a bookshelf",
    "A large dog with its owner in a living room",
    "A man in a white shirt holding a gray dog"
]

# Process with CLIPS
with torch.no_grad(), torch.cuda.amp.autocast() if torch.cuda.is_available() else torch.no_grad():
    # Tokenize all captions
    text_tokens = tokenizer(captions, context_length=model.context_length)
    
    # Encode image and text
    image_features = model.encode_image(image_input)
    text_features = model.encode_text(text_tokens)
    
    # Normalize features
    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)
    
    # Calculate similarity scores (text probabilities)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    
    # Print the results
    print("\nCLIPS Similarity Scores:")
    for i, caption in enumerate(captions):
        print(f"{caption}: {similarity[0][i].item():.2%}")


# # Analysis of CLIP vs. CLIPS Results
# 
# Looking at the similarity scores from both models, there are several interesting patterns and differences worth noting:
# 
# ## Key Observations
# 
# 1. **Different Top Predictions:**
#    - CLIP strongly favors "A gray Great Dane being held by its owner" (78.37%)
#    - CLIPS strongly favors "A man holding a large dog" (76.11%)
# 
# 2. **Specificity vs. Generality:**
#    - CLIP gives higher scores to more specific descriptions (breed identification)
#    - CLIPS gives higher scores to more general but accurate descriptions
# 
# 3. **Secondary Preference:**
#    - CLIP's second choice is "A man in a white shirt holding a gray dog" (13.62%)
#    - CLIPS's second choice is "A gray Great Dane being held by its owner" (16.04%)
# 
# 4. **Similar Rejections:**
#    - Both models correctly assign very low probabilities to obviously incorrect descriptions like "A woman holding a cat" and "A gray horse in a stable"
# 
# 5. **Background Element Recognition:**
#    - Both models assign low probability to "A person standing next to a bookshelf" despite the bookshelf being visible, suggesting they prioritize the main subjects
# 
# ## Model Behavior Analysis
# 
# **CLIP** appears to be more confident in specific visual details like the breed identification (Great Dane) and color attributes. This suggests CLIP may be better at fine-grained visual categorization when trained on web-crawled pairs that often contain specific nomenclature.
# 
# **CLIPS** seems to prioritize the overall scene description and relationship between subjects ("man holding large dog") over specific breed identification. This could reflect its training on synthetic captions that might focus more on relationships and actions rather than specific taxonomic labels.
# 
# ## Practical Implications
# 
# These differences highlight how model training approaches affect what visual-textual relationships are emphasized:
# 
# 1. **Use Case Considerations:**
#    - CLIP might be preferable for applications requiring fine-grained categorization or specific attribute recognition
#    - CLIPS might be better for applications focused on understanding relationships and actions between subjects
# 
# 2. **Caption Style Preferences:**
#    - CLIP appears to favor descriptive, taxonomically precise captions
#    - CLIPS appears to favor action-oriented, relationship-focused captions
# 
# 3. **Error Patterns:**
#    - Both models effectively reject completely incorrect descriptions
#    - Both models assign low probabilities to background elements that are present but not the main focus
# 
# This comparison demonstrates that while both models perform the same fundamental task (visual-language alignment), their different architectures and training approaches result in notably different prioritization of visual information. CLIP's emphasis on specific visual features versus CLIPS's emphasis on relationships and actions highlights the importance of model selection based on specific application requirements.
# 

# 
