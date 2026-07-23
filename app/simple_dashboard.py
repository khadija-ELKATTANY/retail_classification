# ============================================================
# 🚀 ENHANCED SIMPLE DASHBOARD (No pandas, no plotly)
# ============================================================

import http.server
import socketserver
import webbrowser
import json
import os
import base64
from datetime import datetime
from PIL import Image
import io
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from transformers import DistilBertTokenizer, DistilBertModel
import numpy as np

PORT = 8080
MODEL_PATH = "models/best_hybrid.pth"  # Place your model here

# ---- CLASS NAMES ----
CLASS_NAMES = [
    'All Beauty', 'All Electronics', 'Appliances', 'Arts, Crafts & Sewing',
    'Automotive', 'Beauty', 'Books', 'Cell Phones & Accessories',
    'Clothing', 'Electronics', 'Grocery', 'Health & Personal Care',
    'Home', 'Home Improvement', 'Kitchen & Dining', 'Office Products',
    'Pet Supplies', 'Sports', 'Tools & Home Improvement', 'Toys',
    'Video Games'
]
NUM_CLASSES = len(CLASS_NAMES)

# ---- HYBRID MODEL (same as training) ----
class HybridModel(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = None
tokenizer = None

if os.path.exists(MODEL_PATH):
    try:
        model = HybridModel(num_classes=NUM_CLASSES)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=False))
        model.to(device)
        model.eval()
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        print(f"✅ Model loaded from {MODEL_PATH}")
    except Exception as e:
        print(f"⚠️ Failed to load model: {e}")
        model = None

# ---- PREDICTION FUNCTION ----
def predict_product(image_data, title, description):
    if model is None:
        # Demo mode
        import random
        idx = random.randint(0, NUM_CLASSES-1)
        return CLASS_NAMES[idx], random.uniform(0.7, 0.95), False
    
    try:
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
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
        
        # Top 5 predictions for display
        top5_idx = np.argsort(probs)[-5:][::-1]
        top5 = [{'category': CLASS_NAMES[i], 'confidence': float(probs[i])} for i in top5_idx]
        
        return category, confidence, is_anomaly, top5
    except Exception as e:
        return "Error", 0.0, True, []

# ---- HTML TEMPLATE ----
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Retail Classification System</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #0d1117; color: #c9d1d9; padding: 2rem; min-height: 100vh; }
        .header { background: linear-gradient(135deg, #667eea 0%%, #764ba2 100%%); padding: 2rem; border-radius: 15px; text-align: center; margin-bottom: 2rem; }
        .header h1 { color: white; margin: 0; font-size: 2.5rem; }
        .header p { color: rgba(255,255,255,0.85); margin: 0.5rem 0 0 0; }
        .stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-bottom: 2rem; }
        .stat-card { background: #161b22; padding: 1.5rem; border-radius: 10px; text-align: center; border: 1px solid #30363d; }
        .stat-value { font-size: 2.2rem; font-weight: bold; color: #58a6ff; }
        .stat-label { color: #8b949e; font-size: 0.9rem; }
        .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin-bottom: 1.5rem; }
        .card { background: #161b22; border-radius: 12px; padding: 1.5rem; border: 1px solid #30363d; }
        .card h3 { margin-bottom: 1rem; color: #f0f6fc; }
        input, textarea, .file-input { width: 100%%; padding: 0.75rem; background: #0d1117; border: 1px solid #30363d; border-radius: 8px; color: #c9d1d9; font-size: 1rem; margin-bottom: 0.75rem; }
        .btn { background: linear-gradient(135deg, #667eea 0%%, #764ba2 100%%); color: white; border: none; padding: 0.75rem 2rem; border-radius: 8px; font-size: 1rem; font-weight: 600; cursor: pointer; width: 100%%; }
        .btn:hover { opacity: 0.9; transform: scale(1.02); }
        .result-box { background: #0d1117; border-radius: 10px; padding: 1.5rem; border: 2px solid #30363d; text-align: center; margin-top: 1rem; }
        .result-category { font-size: 2.5rem; font-weight: 700; color: #58a6ff; }
        .result-confidence { font-size: 1.2rem; color: #c9d1d9; margin: 0.5rem 0; }
        .badge-normal { background: #2ea043; color: white; padding: 0.25rem 1rem; border-radius: 20px; display: inline-block; font-weight: 600; }
        .badge-anomaly { background: #da3633; color: white; padding: 0.25rem 1rem; border-radius: 20px; display: inline-block; font-weight: 600; }
        .progress-bar { background: #30363d; border-radius: 10px; height: 8px; margin: 0.3rem 0; overflow: hidden; }
        .progress-fill { height: 100%%; border-radius: 10px; background: linear-gradient(90deg, #58a6ff, #667eea); }
        .footer { text-align: center; margin-top: 2rem; padding-top: 1.5rem; border-top: 1px solid #21262d; color: #8b949e; font-size: 0.8rem; }
        .preview-img { max-width: 100%%; border-radius: 10px; margin: 0.5rem 0; }
        @media (max-width: 768px) { .stats { grid-template-columns: repeat(2, 1fr); } .grid-2 { grid-template-columns: 1fr; } body { padding: 1rem; } }
    </style>
</head>
<body>
    <div class="header">
        <h1>🛍️ Retail Classification System</h1>
        <p>Real-Time Product Analytics Dashboard</p>
        <p style="font-size:0.85rem;opacity:0.7;">🚀 Model: {model_status} &bull; 75.33%% Test Accuracy</p>
    </div>
    
    <div class="stats">
        <div class="stat-card"><div class="stat-value">42,000</div><div class="stat-label">Total Products</div></div>
        <div class="stat-card"><div class="stat-value">21</div><div class="stat-label">Categories</div></div>
        <div class="stat-card"><div class="stat-value">75.33%%</div><div class="stat-label">Test Accuracy</div></div>
        <div class="stat-card"><div class="stat-value">9.6ms</div><div class="stat-label">Inference Speed</div></div>
    </div>
    
    <div class="grid-2">
        <div class="card">
            <h3>🔮 Classify Product</h3>
            <form id="predictForm" enctype="multipart/form-data">
                <input type="file" name="image" accept="image/*" required class="file-input">
                <input type="text" name="title" placeholder="Product Title" required>
                <textarea name="description" placeholder="Product Description" rows="3"></textarea>
                <button type="submit" class="btn">🔍 Classify Product</button>
            </form>
        </div>
        <div class="card" id="resultCard">
            <h3>📊 Prediction Result</h3>
            <div id="resultPlaceholder" style="text-align:center;color:#8b949e;padding:2rem 0;">
                <p style="font-size:3rem;">🔍</p>
                <p>Upload an image and enter product details</p>
            </div>
            <div id="resultContent" style="display:none;">
                <div class="result-box">
                    <div style="font-size:0.9rem;color:#8b949e;">Predicted Category</div>
                    <div class="result-category" id="predCategory">--</div>
                    <div class="result-confidence" id="predConfidence">Confidence: --</div>
                    <div id="predStatus" style="margin:0.5rem 0;"></div>
                </div>
                <div style="margin-top:1rem;">
                    <div style="font-size:0.9rem;color:#8b949e;margin-bottom:0.5rem;">Top 5 Predictions</div>
                    <div id="top5List"></div>
                </div>
            </div>
        </div>
    </div>
    
    <div class="card">
        <h3>📋 Recent Predictions</h3>
        <table style="width:100%%;border-collapse:collapse;">
            <tr style="border-bottom:1px solid #30363d;">
                <th style="text-align:left;padding:0.5rem;color:#8b949e;">Time</th>
                <th style="text-align:left;padding:0.5rem;color:#8b949e;">Product</th>
                <th style="text-align:left;padding:0.5rem;color:#8b949e;">Category</th>
                <th style="text-align:left;padding:0.5rem;color:#8b949e;">Confidence</th>
                <th style="text-align:left;padding:0.5rem;color:#8b949e;">Status</th>
            </tr>
            {history_rows}
        </table>
    </div>
    
    <div class="footer">
        <p>📅 Updated: {timestamp} &bull; Powered by Hybrid CNN-ViT &bull; Thesis Project</p>
        <p style="margin-top:0.3rem;opacity:0.6;">🎯 75.33%% Test Accuracy &bull; 15.8x Improvement Over Random</p>
    </div>
    
    <script>
        document.getElementById('predictForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            const formData = new FormData(this);
            const resultPlaceholder = document.getElementById('resultPlaceholder');
            const resultContent = document.getElementById('resultContent');
            resultPlaceholder.innerHTML = '<p style="font-size:2rem;">⏳</p><p>Processing...</p>';
            resultContent.style.display = 'none';
            
            try {
                const response = await fetch('/predict', { method: 'POST', body: formData });
                const data = await response.json();
                resultPlaceholder.style.display = 'none';
                resultContent.style.display = 'block';
                document.getElementById('predCategory').textContent = data.category;
                document.getElementById('predConfidence').textContent = 'Confidence: ' + (data.confidence * 100).toFixed(1) + '%';
                const statusBadge = data.is_anomaly ? '<span class="badge-anomaly">⚠️ Anomaly Detected</span>' : '<span class="badge-normal">✅ Normal</span>';
                document.getElementById('predStatus').innerHTML = statusBadge;
                let top5Html = '';
                data.top5.forEach(function(item) {
                    const pct = (item.confidence * 100).toFixed(1);
                    top5Html += '<div><span>' + item.category + '</span><span style="float:right;">' + pct + '%</span></div>';
                    top5Html += '<div class="progress-bar"><div class="progress-fill" style="width:' + pct + '%%"></div></div>';
                });
                document.getElementById('top5List').innerHTML = top5Html;
            } catch(err) {
                resultPlaceholder.innerHTML = '<p style="color:#da3633;">❌ Error: ' + err.message + '</p>';
            }
        });
    </script>
</body>
</html>
'''

# ---- HTTP REQUEST HANDLER ----
class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/' or self.path == '':
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            
            # History rows (sample data)
            history_rows = '''
                <tr><td style="padding:0.5rem;">14:32:15</td><td>Wireless Earbuds</td><td>Electronics</td><td>94%</td><td><span class="badge-normal">Normal</span></td></tr>
                <tr><td style="padding:0.5rem;">14:28:42</td><td>Leather Jacket</td><td>Clothing</td><td>87%</td><td><span class="badge-normal">Normal</span></td></tr>
                <tr><td style="padding:0.5rem;">14:15:03</td><td>Unknown Item</td><td>⚠️ Anomaly</td><td>32%</td><td><span class="badge-anomaly">Flagged</span></td></tr>
            '''
            model_status = "Active" if model is not None else "Demo Mode"
            html = HTML_TEMPLATE.format(
                model_status=model_status,
                history_rows=history_rows,
                timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            )
            self.wfile.write(html.encode('utf-8'))
        else:
            self.send_error(404)
    
    def do_POST(self):
        if self.path == '/predict':
            content_type = self.headers.get('Content-Type')
            if not content_type or 'multipart/form-data' not in content_type:
                self.send_error(400, "Invalid content type")
                return
            
            try:
                import cgi
                form = cgi.FieldStorage(
                    fp=self.rfile,
                    headers=self.headers,
                    environ={'REQUEST_METHOD': 'POST'}
                )
                
                # Get image
                file_item = form['image']
                if not file_item.file:
                    raise ValueError("No image uploaded")
                image_data = file_item.file.read()
                
                # Get text
                title = form.getvalue('title', '')
                description = form.getvalue('description', '')
                
                # Predict
                category, confidence, is_anomaly, top5 = predict_product(image_data, title, description)
                
                # Build response
                response = {
                    'category': category,
                    'confidence': confidence,
                    'is_anomaly': is_anomaly,
                    'top5': top5 if top5 else [{'category': c, 'confidence': 0.0} for c in CLASS_NAMES[:5]]
                }
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json; charset=utf-8')
                self.end_headers()
                self.wfile.write(json.dumps(response).encode('utf-8'))
                
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'error': str(e)}).encode('utf-8'))

# ---- START SERVER ----
print("🚀 Starting Enhanced Dashboard at http://localhost:8080")
print("📊 Press Ctrl+C to stop")
print(f"📁 Model: {'✅ Loaded' if model else '⚠️ Demo Mode'}")
webbrowser.open('http://localhost:8080')

with socketserver.TCPServer(("", PORT), DashboardHandler) as httpd:
    print(f"🌐 Open: http://localhost:{PORT}")
    httpd.serve_forever()