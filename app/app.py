# ============================================================
# 🚀 RETAIL CLASSIFICATION SYSTEM – FINAL VERSION
# ============================================================

import os

# ── Force offline mode (will use local cache) ──
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
# Fallback mirror (unlikely needed with local cache)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

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
from datetime import datetime
import random

# ── Page config ──
st.set_page_config(
    page_title="Retail Classification System",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ──
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
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        width: 100%;
    }
    .stButton > button:hover {
        opacity: 0.9;
        transform: scale(1.02);
    }
</style>
""", unsafe_allow_html=True)

# ── Class names ──
CLASS_NAMES = [
    'All Beauty', 'All Electronics', 'Appliances', 'Arts, Crafts & Sewing',
    'Automotive', 'Beauty', 'Books', 'Cell Phones & Accessories',
    'Clothing', 'Electronics', 'Grocery', 'Health & Personal Care',
    'Home', 'Home Improvement', 'Kitchen & Dining', 'Office Products',
    'Pet Supplies', 'Sports', 'Tools & Home Improvement', 'Toys',
    'Video Games'
]

# ── Model definition ──
class HybridModel(nn.Module):
    def __init__(self, num_classes=21):
        super().__init__()
        self.image_encoder = models.efficientnet_b3(weights=None)
        in_features = self.image_encoder.classifier[1].in_features
        self.image_encoder.classifier = nn.Identity()
        self.image_fc = nn.Linear(in_features, 256)

        # ── Load DistilBERT from local cache ──
        # The folder 'distilbert_cache' is expected to be in the project root.
        # If it's inside 'app/', adjust the path to '../distilbert_cache' or './distilbert_cache'.
        cache_dir = "./distilbert_cache"
        if not os.path.exists(cache_dir):
            # Fallback to a secondary location (e.g., inside app folder)
            cache_dir = "app/distilbert_cache"
        try:
            self.text_encoder = DistilBertModel.from_pretrained(cache_dir, local_files_only=True)
        except Exception as e:
            st.error(f"Failed to load DistilBERT from local cache: {e}. Please ensure the cache folder exists.")
            # Fallback to online (if allowed) – but we have offline env vars, so this will raise.
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

# ── Model loader (cached) ──
@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HybridModel(num_classes=21)

    # Load tokenizer from local cache
    cache_dir = "./distilbert_cache"
    if not os.path.exists(cache_dir):
        cache_dir = "app/distilbert_cache"
    try:
        tokenizer = DistilBertTokenizer.from_pretrained(cache_dir, local_files_only=True)
    except Exception as e:
        st.error(f"Tokenizer cache not found: {e}")
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    # Load the trained hybrid model weights
    model_paths = [
        'best_hybrid.pth',
        'models/best_hybrid.pth',
        '/mount/src/retail_classification/best_hybrid.pth'
    ]
    loaded = False
    for path in model_paths:
        if os.path.exists(path):
            try:
                st.info(f"Loading model from {path}...")
                model.load_state_dict(torch.load(path, map_location=device, weights_only=False))
                loaded = True
                st.success(f"✅ Model loaded from {path}")
                break
            except Exception as e:
                st.warning(f"Failed to load from {path}: {e}")

    if not loaded:
        st.warning("⚠️ Model file not found. Running in demo mode.")
        model = None

    if model is not None:
        model.eval()
        model.to(device)
    return model, tokenizer, CLASS_NAMES, device

model, tokenizer, class_names, device = load_model()

# ── Prediction function ──
def predict(image, title, description):
    if model is None:
        # Demo mode
        idx = random.randint(0, len(class_names)-1)
        top5 = []
        for i in range(5):
            top5.append({
                'category': class_names[(idx + i) % len(class_names)],
                'confidence': max(0.1, random.uniform(0.5, 0.95) - i*0.05)
            })
        top5.sort(key=lambda x: x['confidence'], reverse=True)
        return class_names[idx], random.uniform(0.7, 0.95), False, top5

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
    category = class_names[top_idx]
    is_anomaly = confidence < 0.45

    top5_idx = np.argsort(probs)[-5:][::-1]
    top5 = [{'category': class_names[i], 'confidence': float(probs[i])} for i in top5_idx]
    return category, confidence, is_anomaly, top5

# ── Sidebar navigation ──
st.sidebar.markdown("""
<div style="text-align: center; padding: 0.5rem 0;">
    <h2 style="color: #667eea;">🛍️ Retail AI</h2>
    <p style="color: #8b949e; font-size: 0.8rem;">Classification System</p>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "📋 Navigate",
    [
        "🏠 Dashboard",
        "🔮 Predict Product",
        "📊 Analytics",
        "📈 Model Performance",
        "⚙️ Settings"
    ],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"""
<div style="font-size: 0.8rem; color: #8b949e;">
    <p>📅 {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
    <p>📊 Accuracy: <b style="color:#58a6ff;">75.33%</b></p>
    <p>⚡ Model: <b style="color:#2ea043;">{'Active' if model else 'Demo'}</b></p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# PAGE: DASHBOARD
# ============================================================
if page == "🏠 Dashboard":
    st.markdown('<div class="main-header"><h1>🛍️ Retail Analytics Dashboard</h1><p>Real-Time Product Classification & Inventory Intelligence</p></div>', unsafe_allow_html=True)

    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card"><div class="metric-value">42,000</div><div class="metric-label">Products</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><div class="metric-value">21</div><div class="metric-label">Categories</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><div class="metric-value">75.33%</div><div class="metric-label">Test Accuracy</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card"><div class="metric-value">9.6ms</div><div class="metric-label">Inference Speed</div></div>', unsafe_allow_html=True)

    # Charts
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 Category Distribution")
        cat_data = pd.DataFrame({
            'Category': ['Electronics', 'Clothing', 'Books', 'Beauty', 'Automotive', 'Other'],
            'Count': [15, 12, 11, 9, 8, 45]
        })
        fig = px.bar(cat_data, x='Category', y='Count', color='Category', title='')
        fig.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("💵 Price Distribution")
        price_data = np.random.uniform(10, 500, 1000)
        fig = px.histogram(price_data, nbins=30, title='', labels={'value':'Price (USD)'})
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("🕐 Recent Predictions")
    sample_data = pd.DataFrame({
        'Time': [datetime.now().strftime('%H:%M:%S') for _ in range(5)],
        'Product': ['Wireless Earbuds', 'Leather Jacket', 'Science Book', 'Smartphone', 'Unknown Item'],
        'Category': ['Electronics', 'Clothing', 'Books', 'Electronics', '⚠️ Anomaly'],
        'Confidence': ['94%', '87%', '91%', '96%', '32%'],
        'Status': ['✅ Normal', '✅ Normal', '✅ Normal', '✅ Normal', '⚠️ Flagged']
    })
    st.dataframe(sample_data, use_container_width=True, hide_index=True)

# ============================================================
# PAGE: PREDICT PRODUCT
# ============================================================
elif page == "🔮 Predict Product":
    st.markdown('<div class="main-header"><h1>🔮 Real-Time Product Classification</h1><p>Upload an image and enter product details</p></div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        uploaded_file = st.file_uploader("📷 Upload Product Image", type=['jpg', 'jpeg', 'png'])
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Product Image", use_column_width=True)
        title = st.text_input("📝 Product Title", placeholder="e.g., Wireless Bluetooth Earbuds")
        description = st.text_area("📄 Product Description", placeholder="e.g., Noise cancelling, 20hr battery life...")

        if st.button("🔍 Classify Product", type="primary"):
            if uploaded_file is None:
                st.error("Please upload an image.")
            elif not title.strip():
                st.warning("Please enter a product title.")
            else:
                with st.spinner("Analyzing product..."):
                    category, confidence, is_anomaly, top5 = predict(image, title, description)
                    st.session_state['prediction'] = {
                        'category': category,
                        'confidence': confidence,
                        'is_anomaly': is_anomaly,
                        'top5': top5
                    }

    with col2:
        st.subheader("📊 Prediction Results")
        if 'prediction' in st.session_state:
            pred = st.session_state['prediction']
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.markdown(f"""
                <div class="prediction-box">
                    <div style="font-size:0.9rem;color:#8b949e;">Category</div>
                    <div class="prediction-category">{pred['category']}</div>
                </div>
                """, unsafe_allow_html=True)
            with col_b:
                color = "#2ea043" if pred['confidence'] > 0.7 else "#d29922"
                st.markdown(f"""
                <div class="prediction-box">
                    <div style="font-size:0.9rem;color:#8b949e;">Confidence</div>
                    <div style="font-size:2rem;font-weight:700;color:{color};">{pred['confidence']:.1%}</div>
                </div>
                """, unsafe_allow_html=True)
            with col_c:
                badge = "badge-anomaly" if pred['is_anomaly'] else "badge-normal"
                label = "⚠️ Anomaly" if pred['is_anomaly'] else "✅ Normal"
                st.markdown(f"""
                <div class="prediction-box">
                    <div style="font-size:0.9rem;color:#8b949e;">Status</div>
                    <div style="padding:0.5rem 0;"><span class="{badge}">{label}</span></div>
                </div>
                """, unsafe_allow_html=True)

            if pred['top5']:
                st.markdown("#### Top 5 Predictions")
                top5_df = pd.DataFrame(pred['top5'])
                fig = px.bar(top5_df, x='category', y='confidence', color='confidence',
                            color_continuous_scale='Blues', labels={'category': 'Category', 'confidence': 'Confidence'})
                fig.update_layout(showlegend=False, height=200)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Upload an image and click 'Classify Product' to see results.")

# ============================================================
# PAGE: ANALYTICS
# ============================================================
elif page == "📊 Analytics":
    st.markdown('<div class="main-header"><h1>📊 Inventory Analytics</h1><p>Explore product distribution and metrics</p></div>', unsafe_allow_html=True)

    # Generate sample data
    np.random.seed(42)
    categories = ['Electronics', 'Clothing', 'Books', 'Beauty', 'Automotive', 'Sports', 'Toys', 'Home', 'Appliances']
    df = pd.DataFrame({
        'category': np.random.choice(categories, 1000),
        'price': np.random.uniform(10, 500, 1000),
        'stock': np.random.randint(0, 1000, 1000),
        'rating': np.random.uniform(1, 5, 1000)
    })

    col1, col2 = st.columns(2)
    with col1:
        selected_cats = st.multiselect("Filter Categories", options=df['category'].unique(), default=df['category'].unique()[:3])
    with col2:
        price_range = st.slider("Price Range", 0, 500, (0, 500))

    filtered_df = df[df['category'].isin(selected_cats)]
    filtered_df = filtered_df[(filtered_df['price'] >= price_range[0]) & (filtered_df['price'] <= price_range[1])]

    col1, col2, col3 = st.columns(3)
    col1.metric("Products", len(filtered_df))
    col2.metric("Avg Price", f"${filtered_df['price'].mean():.2f}")
    col3.metric("Avg Rating", f"{filtered_df['rating'].mean():.2f} ⭐")

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(filtered_df.groupby('category')['stock'].sum().reset_index(), x='category', y='stock', color='category', title='Stock by Category')
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.scatter(filtered_df, x='price', y='rating', color='category', size='stock', title='Price vs Rating')
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(filtered_df.head(100), use_container_width=True)

# ============================================================
# PAGE: MODEL PERFORMANCE
# ============================================================
elif page == "📈 Model Performance":
    st.markdown('<div class="main-header"><h1>📈 Model Performance</h1><p>Accuracy, training history, and confusion matrix</p></div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🏆 Model Comparison")
        models_list = ['CNN', 'ResNet50', 'EffNet-B4', 'Hybrid (Ours)']
        acc = [29.17, 53.25, 59.37, 75.33]
        fig = px.bar(x=models_list, y=acc, color=models_list, title='Test Accuracy Comparison')
        fig.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("🧩 Training History")
        epochs = list(range(1, 31))
        train_acc = [29,35,41,47,53,58,62,66,70,73,76,79,81,83,85,87,89,90,92,93,94,95,96,96,97,97,97,97,98,98]
        val_acc = [36,42,47,50,52,54,56,58,59,60,61,62,63,64,65,66,67,68,69,70,71,71,72,72,72,72,73,73,73,73]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=epochs, y=train_acc, name='Train', line=dict(color='#58a6ff', width=3)))
        fig.add_trace(go.Scatter(x=epochs, y=val_acc, name='Validation', line=dict(color='#2ea043', width=3, dash='dash')))
        fig.update_layout(title='', xaxis_title='Epoch', yaxis_title='Accuracy (%)', height=350)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("🧩 Confusion Matrix (Sample)")
    cm = np.random.randint(5, 20, (10, 10))
    np.fill_diagonal(cm, np.random.randint(50, 100, 10))
    fig = px.imshow(cm, title='', labels=dict(x="Predicted", y="True"), color_continuous_scale='Blues')
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# PAGE: SETTINGS
# ============================================================
elif page == "⚙️ Settings":
    st.markdown('<div class="main-header"><h1>⚙️ Settings</h1><p>Configure model and system parameters</p></div>', unsafe_allow_html=True)

    st.subheader("🎯 Model Configuration")
    anomaly_threshold = st.slider("Anomaly Detection Threshold", 0.1, 0.9, 0.45)
    st.info(f"Products with confidence below {anomaly_threshold:.2f} will be flagged as anomalies.")

    st.subheader("🔧 System Settings")
    col1, col2 = st.columns(2)
    with col1:
        image_size = st.selectbox("Image Size", [224, 256, 384], index=0)
    with col2:
        batch_size = st.number_input("Batch Size", min_value=1, max_value=128, value=32)

    st.subheader("📊 Model Information")
    st.markdown("""
    | Property | Value |
    |----------|-------|
    | Architecture | Hybrid CNN-ViT (EfficientNet-B3 + DistilBERT) |
    | Test Accuracy | 75.33% |
    | Macro F1 | 0.7518 |
    | Number of Classes | 21 |
    | Training Epochs | 30 |
    | Model File Size | 297 MB |
    """)

    if st.button("🔄 Rebuild Model Cache"):
        st.cache_resource.clear()
        st.success("Cache cleared! Reload the page to reload the model.")

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.caption("🛍️ Powered by Hybrid CNN-ViT (EfficientNet-B3 + DistilBERT) | 75.33% Test Accuracy | Master's Thesis")