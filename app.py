import torch

from PIL import Image

import streamlit as st

from transformers import BlipProcessor, BlipForConditionalGeneration

st.set_page_config(page_title="BLIP Image Captioning", page_icon="🖼️", layout="centered")

@st.cache_resource

def load_models():

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

device = "cuda" if torch.cuda.is_available() else "cpu"

model.to(device)

return processor, model, device

st.title("Image Captioning with BLIP")

processor, model, device = load_models()

up = st.file_uploader("Upload an image", type=["png","jpg","jpeg"])

if up:

img = Image.open(up).convert("RGB")

st.image(img, caption="Input", use_column_width=True)

with st.spinner("Generating..."):

inputs = processor(img, return_tensors="pt").to(device)

out = model.generate(**inputs)

caption = processor.decode(out, skip_special_tokens=True)

st.success(f"Caption: {caption}")
