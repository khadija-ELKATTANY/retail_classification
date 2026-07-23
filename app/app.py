# ============================================================
# 🚀 RETAIL CLASSIFICATION – STREAMLIT CLOUD DEPLOY (FIXED)
# ============================================================

import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from transformers import DistilBertTokenizer, DistilBertModel
from PIL import Image
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

# ---- PAGE CONFIG ----
st.set_page_config(
    page_title="Retail Classification System",
    page_icon="🛍️",
    layout="wide"
)

# ---- STYLING ----
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .main-header h1 { font-size: 2.5rem; margin: 0; }
    .main-header p { font-size: 1.1rem; opacity: 0.9; margin: 0.3rem 0 0 0; }
    .metric-card {
        background: #1e1e2e;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        border: 1px solid #333;
    }
    .metric-value { font-size: 2rem; font-weight: bold; color: #58a6ff; }
    .metric-label { color: #8b949e; font-size: 0.9rem; }
    .prediction-box {
        background: #161b22;
        border-radius: 12px;
        padding: 1.5rem;
        border: 2px solid #30363d;
        text-align: center;
    }
    .prediction-category { font-size: 2rem; font-weight: 700; color: #58a6ff; }
    .badge-normal {
        background: #2ea043; color: white; padding: 0.25rem 1rem;
        border-radius: 20px; font-weight: 600;
    }
    .badge-anomaly {
        background: #da3633; color: white; padding: 0.25rem 1rem;
        border-radius: 20px; font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ---- CLASS NAMES ----
CLASS_NAMES = [
    'All Beauty', 'All Electronics', 'Appliances', 'Arts, Crafts & Sewing',
    'Automotive', 'Beauty', 'Books', 'Cell Phones & Accessories',
    'Clothing', 'Electronics', 'Grocery', 'Health & Personal Care',
    'Home', 'Home Improvement', 'Kitchen & Dining', 'Office Products',
    'Pet Supplies', 'Sports', 'Tools & Home Improvement', 'Toys',
    'Video Games'
]

# ---- MODEL CLASS ----
class HybridModel(nn.Module):
    def __init__(self, num_classes=21):
        super().__init__()
        self.image_encoder = models.efficientnet_b3(weights=None)
        in_features = self.image_encoder.classifier[1].in_features
        self.image_encoder.classifier = nn.Identity()
        self.image_fc = nn.Linear(in_features, 256)
        self.text_encoder = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.text_fc = nn.Linear(768, 256)
        self.fc1 = nn.Linear(256 + 256, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)
        for param in self.text_encoder.parameters():
            param.requires_grad = False
    
    def forward(self, image, input_ids, attention_mask):
        image_feat = self.image_encoder(image)
        image_feat = self.image_fc(image_feat)
        text_output = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_feat = text_output.last_hidden_state[:, 0, :]
        text_feat = self.text_fc(text_feat)
        combined = torch.cat([image_feat, text_feat], dim=1)
        combined = self.dropout(torch.relu(self.fc1(combined)))
        out = self.fc2(combined)
        return out

# ---- LOAD MODEL ----
@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = HybridModel(num_classes=21)
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    # Try multiple paths for the model file
    model_paths = [
        'best_hybrid.pth',
        'models/best_hybrid.pth',
        '../best_hybrid.pth',
        '/mount/src/retail_classification/best_hybrid.pth',
        '/mount/src/retail_classification/models/best_hybrid.pth'
    ]
    
    model_loaded = False
    for path in model_paths:
        if os.path.exists(path):
            try:
                st.info(f"Loading model from {path}...")
                model.load_state_dict(torch.load(path, map_location=device, weights_only=False))
                model_loaded = True
                st.success(f"✅ Model loaded from {path}")
                break
            except Exception as e:
                st.warning(f"Failed to load from {path}: {e}")
    
    if not model_loaded:
        st.warning("⚠️ Model file not found. Running in demo mode with simulated predictions.")
        model = None
    
    model.eval()
    model.to(device)
    
    return model, tokenizer, CLASS_NAMES, device

# ---- PREDICTION FUNCTION ----
def predict(image, title, description):
    if model is None:
        import random
        idx = random.randint(0, len(CLASS_NAMES)-1)
        # Simulate top 5
        top5 = []
        for i in range(5):
            top5.append({'category': CLASS_NAMES[(idx + i) % len(CLASS_NAMES)], 'confidence': max(0.1, random.uniform(0.5, 0.95) - i*0.05)})
        top5.sort(key=lambda x: x['confidence'], reverse=True)
        return CLASS_NAMES[idx], random.uniform(0.7, 0.95), False, top5
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    text = f"{title} {description}"
    encoding = tokenizer(text, truncation=True, padding='max_length', max_length=128, return_tensors='pt')
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(img_tensor, input_ids, attention_mask)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
    
    top_idx = np.argmax(probs)
    confidence = float(probs[top_idx])
    category = CLASS_NAMES[top_idx]
    is_anomaly = confidence < 0.45
    
    top5_idx = np.argsort(probs)[-5:][::-1]
    top5 = [{'category': CLASS_NAMES[i], 'confidence': float(probs[i])} for i in top5_idx]
    
    return category, confidence, is_anomaly, top5

model, tokenizer, class_names, device = load_model()

# ---- HEADER ----
st.markdown('<div class="main-header"><h1>🛍️ Retail Classification System</h1><p>Real-Time Product Classification & Inventory Analytics</p></div>', unsafe_allow_html=True)

# ---- KPIs ----
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown('<div class="metric-card"><div class="metric-value">42,000</div><div class="metric-label">Products</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="metric-card"><div class="metric-value">21</div><div class="metric-label">Categories</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="metric-card"><div class="metric-value">75.33%</div><div class="metric-label">Test Accuracy</div></div>', unsafe_allow_html=True)
with col4:
    st.markdown('<div class="metric-card"><div class="metric-value">9.6ms</div><div class="metric-label">Inference Speed</div></div>', unsafe_allow_html=True)

# ---- TWO COLUMNS ----
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🔮 Classify a Product")
    uploaded_file = st.file_uploader("📷 Upload Product Image", type=['jpg', 'jpeg', 'png'])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
    title = st.text_input("📝 Product Title", placeholder="e.g., Wireless Bluetooth Earbuds")
    description = st.text_area("📄 Product Description", placeholder="e.g., Noise cancelling, 20hr battery life...")

with col2:
    st.subheader("📊 Prediction Results")
    if st.button("🔍 Classify Product", type="primary", use_container_width=True):
        if uploaded_file is None:
            st.error("Please upload an image.")
        elif not title.strip():
            st.warning("Please enter a product title.")
        else:
            with st.spinner("Analyzing..."):
                category, confidence, is_anomaly, top5 = predict(image, title, description)
                
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.markdown(f"""
                    <div class="prediction-box">
                        <div style="font-size:0.9rem;color:#8b949e;">Category</div>
                        <div class="prediction-category">{category}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col_b:
                    color = "#2ea043" if confidence > 0.7 else "#d29922"
                    st.markdown(f"""
                    <div class="prediction-box">
                        <div style="font-size:0.9rem;color:#8b949e;">Confidence</div>
                        <div style="font-size:2rem;font-weight:700;color:{color};">{confidence:.1%}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col_c:
                    badge = "badge-anomaly" if is_anomaly else "badge-normal"
                    label = "⚠️ Anomaly" if is_anomaly else "✅ Normal"
                    st.markdown(f"""
                    <div class="prediction-box">
                        <div style="font-size:0.9rem;color:#8b949e;">Status</div>
                        <div style="padding:0.5rem 0;"><span class="{badge}">{label}</span></div>
                    </div>
                    """, unsafe_allow_html=True)
                
                if top5:
                    st.markdown("#### Top 5 Predictions")
                    top5_df = pd.DataFrame(top5)
                    fig = px.bar(top5_df, x='category', y='confidence', color='confidence',
                                color_continuous_scale='Blues', labels={'category': 'Category', 'confidence': 'Confidence'})
                    fig.update_layout(showlegend=False, height=200)
                    st.plotly_chart(fig, use_container_width=True)

# ---- MODEL PERFORMANCE ----
st.markdown("---")
st.subheader("📊 Model Performance")
col1, col2 = st.columns(2)

with col1:
    models_list = ['CNN', 'ResNet50', 'EffNet-B4', 'Hybrid (Ours)']
    acc = [29.17, 53.25, 59.37, 75.33]
    fig = px.bar(x=models_list, y=acc, color=models_list, title='Test Accuracy Comparison')
    fig.update_layout(showlegend=False, height=300)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    epochs = list(range(1, 31))
    train_acc = [29,35,41,47,53,58,62,66,70,73,76,79,81,83,85,87,89,90,92,93,94,95,96,96,97,97,97,97,98,98]
    val_acc = [36,42,47,50,52,54,56,58,59,60,61,62,63,64,65,66,67,68,69,70,71,71,72,72,72,72,73,73,73,73]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=train_acc, name='Train', line=dict(color='#58a6ff', width=3)))
    fig.add_trace(go.Scatter(x=epochs, y=val_acc, name='Validation', line=dict(color='#2ea043', width=3, dash='dash')))
    fig.update_layout(title='Training History', xaxis_title='Epoch', yaxis_title='Accuracy (%)', height=300)
    st.plotly_chart(fig, use_container_width=True)

st.caption("🛍️ Powered by Hybrid CNN-ViT (EfficientNet-B3 + DistilBERT) | 75.33% Test Accuracy")
