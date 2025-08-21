import streamlit as st
from fastai.vision.all import load_learner, PILImage, FocalLossFlat
import torch
import numpy as np
import pandas as pd

# -----------------------------
# 1️⃣ Load models
# -----------------------------
@st.cache_resource(show_spinner=True)
def load_models():
    resnet_model = load_learner("models/ham_resnet50.pkl")
    focalloss_model = load_learner("models/ham_resnet50_focalloss.pkl")
    focalloss_model.loss_func = FocalLossFlat()  # fix NV collapse
    return resnet_model, focalloss_model

resnet_model, focalloss_model = load_models()
st.success("✅ Models loaded successfully!")

# -----------------------------
# 2️⃣ Preprocess image
# -----------------------------
def preprocess_image(img: PILImage):
    img_array = np.array(img).astype(np.float32)/255.0
    img_tensor = torch.tensor(img_array, dtype=torch.float32).permute(2,0,1)
    mean = torch.tensor([0.485,0.456,0.406])
    std  = torch.tensor([0.229,0.224,0.225])
    img_tensor = (img_tensor - mean[:,None,None])/std[:,None,None]
    return img_tensor.unsqueeze(0)

# -----------------------------
# 3️⃣ Prediction function
# -----------------------------
def predict(img_path):
    img = PILImage.create(img_path).convert("RGB")
    img_tensor = preprocess_image(img)

    # ResNet50
    with torch.no_grad():
        preds = resnet_model.model(img_tensor)
        probs = torch.nn.functional.softmax(preds[0], dim=0)
        resnet_idx = torch.argmax(probs).item()
        resnet_class = resnet_model.dls.vocab[resnet_idx]
        resnet_conf = probs[resnet_idx].item() * 100
        resnet_probs = {resnet_model.dls.vocab[i]: float(probs[i]) for i in range(len(probs))}

    # ResNet50 + FocalLoss
    with torch.no_grad():
        preds = focalloss_model.model(img_tensor)
        probs = torch.nn.functional.softmax(preds[0], dim=0)
        focal_idx = torch.argmax(probs).item()
        focal_class = focalloss_model.dls.vocab[focal_idx]
        focal_conf = probs[focal_idx].item() * 100
        focal_probs = {focalloss_model.dls.vocab[i]: float(probs[i]) for i in range(len(probs))}

    return img, (resnet_class, resnet_conf, resnet_probs), (focal_class, focal_conf, focal_probs)

# -----------------------------
# 4️⃣ Streamlit interface
# -----------------------------
st.title("🩺 Skin Cancer Detection: ResNet vs FocalLoss")
st.write("Upload a dermatoscopic image to get predictions from both models.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg","jpeg","png"])
if uploaded_file is not None:
    img, resnet_result, focal_result = predict(uploaded_file)

    # Display image
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Display top predictions
    st.subheader("Predictions")
    st.write(f"**ResNet50:** {resnet_result[0]} ({resnet_result[1]:.2f}% confidence)")
    st.write(f"**ResNet50 + FocalLoss:** {focal_result[0]} ({focal_result[1]:.2f}% confidence)")

    # Display full probability table
    st.subheader("All Class Probabilities")
    prob_df = pd.DataFrame({
        "Class": list(resnet_result[2].keys()),
        "ResNet50": [v*100 for v in resnet_result[2].values()],
        "ResNet50 + FocalLoss": [v*100 for v in focal_result[2].values()]
    })
    st.table(prob_df)
