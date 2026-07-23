# ============================================================
# 🚀 RETAIL DASHBOARD – DEMO MODE (No PyTorch Required)
# ============================================================

import http.server
import socketserver
import webbrowser
from datetime import datetime

PORT = 8080

HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Retail Classification Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #0d1117; color: #c9d1d9; padding: 2rem; min-height: 100vh; }
        .header { background: linear-gradient(135deg, #667eea 0%%, #764ba2 100%%); padding: 2rem; border-radius: 15px; text-align: center; margin-bottom: 2rem; }
        .header h1 { color: white; margin: 0; font-size: 2.5rem; }
        .header p { color: rgba(255,255,255,0.85); margin: 0.5rem 0 0 0; }
        .header .sub { font-size: 0.85rem; opacity: 0.7; margin-top: 0.5rem; }
        .stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-bottom: 2rem; }
        .stat-card { background: #161b22; padding: 1.5rem; border-radius: 10px; text-align: center; border: 1px solid #30363d; }
        .stat-value { font-size: 2.2rem; font-weight: bold; color: #58a6ff; }
        .stat-label { color: #8b949e; font-size: 0.9rem; }
        .grid-2 { display: grid; grid-template-columns: 2fr 1fr; gap: 1.5rem; }
        .card { background: #161b22; border-radius: 12px; padding: 1.5rem; border: 1px solid #30363d; }
        .card h3 { margin-bottom: 1rem; color: #f0f6fc; font-weight: 500; }
        table { width: 100%%; border-collapse: collapse; }
        th { text-align: left; padding: 0.6rem 0.5rem; border-bottom: 1px solid #30363d; color: #8b949e; font-weight: 500; font-size: 0.85rem; text-transform: uppercase; }
        td { padding: 0.6rem 0.5rem; border-bottom: 1px solid #21262d; }
        .badge-success { background: #2ea043; color: white; padding: 0.2rem 0.8rem; border-radius: 20px; font-size: 0.75rem; font-weight: 600; display: inline-block; }
        .badge-danger { background: #da3633; color: white; padding: 0.2rem 0.8rem; border-radius: 20px; font-size: 0.75rem; font-weight: 600; display: inline-block; }
        .progress-bar { background: #30363d; border-radius: 10px; height: 8px; margin: 0.3rem 0; overflow: hidden; }
        .progress-fill { height: 100%%; border-radius: 10px; background: linear-gradient(90deg, #58a6ff, #667eea); }
        .category-row { display: flex; justify-content: space-between; align-items: center; padding: 0.3rem 0; }
        .footer { text-align: center; margin-top: 2rem; padding-top: 1.5rem; border-top: 1px solid #21262d; color: #8b949e; font-size: 0.8rem; }
        @media (max-width: 768px) { .stats { grid-template-columns: repeat(2, 1fr); } .grid-2 { grid-template-columns: 1fr; } body { padding: 1rem; } }
        .demo-badge { background: #d29922; color: white; padding: 0.2rem 0.8rem; border-radius: 20px; font-size: 0.7rem; display: inline-block; margin-left: 0.5rem; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🛍️ Retail Classification System</h1>
        <p>Real-Time Product Analytics Dashboard</p>
        <div class="sub">📊 Demo Mode (Model Results from Kaggle) &bull; 🚀 75.33%% Test Accuracy &bull; ⚡ 9.6ms Inference</div>
    </div>
    
    <div class="stats">
        <div class="stat-card"><div class="stat-value">42,000</div><div class="stat-label">Products</div></div>
        <div class="stat-card"><div class="stat-value">21</div><div class="stat-label">Categories</div></div>
        <div class="stat-card"><div class="stat-value">75.33%%</div><div class="stat-label">Test Accuracy</div></div>
        <div class="stat-card"><div class="stat-value">9.6ms</div><div class="stat-label">Inference Speed</div></div>
    </div>
    
    <div class="grid-2">
        <div class="card">
            <h3>📋 Recent Predictions <span class="demo-badge">Demo</span></h3>
            <table>
                <thead><tr><th>Product</th><th>Category</th><th>Confidence</th><th>Status</th></tr></thead>
                <tbody>
                    <tr><td>Wireless Earbuds</td><td>Electronics</td><td>94%%</td><td><span class="badge-success">Normal</span></td></tr>
                    <tr><td>Leather Jacket</td><td>Clothing</td><td>87%%</td><td><span class="badge-success">Normal</span></td></tr>
                    <tr><td>Science Book</td><td>Books</td><td>91%%</td><td><span class="badge-success">Normal</span></td></tr>
                    <tr><td>Smartphone</td><td>Electronics</td><td>96%%</td><td><span class="badge-success">Normal</span></td></tr>
                    <tr><td>Unknown Item</td><td>⚠️ Anomaly</td><td>32%%</td><td><span class="badge-danger">Flagged</span></td></tr>
                </tbody>
            </table>
        </div>
        <div class="card">
            <h3>📊 Category Breakdown</h3>
            <div class="category-row"><span>Electronics</span><span>15%%</span></div>
            <div class="progress-bar"><div class="progress-fill" style="width:15%%"></div></div>
            <div class="category-row"><span>Clothing</span><span>12%%</span></div>
            <div class="progress-bar"><div class="progress-fill" style="width:12%%"></div></div>
            <div class="category-row"><span>Books</span><span>11%%</span></div>
            <div class="progress-bar"><div class="progress-fill" style="width:11%%"></div></div>
            <div class="category-row"><span>Beauty</span><span>9%%</span></div>
            <div class="progress-bar"><div class="progress-fill" style="width:9%%"></div></div>
            <div class="category-row"><span>Automotive</span><span>8%%</span></div>
            <div class="progress-bar"><div class="progress-fill" style="width:8%%"></div></div>
            <div class="category-row"><span>Other (16)</span><span>45%%</span></div>
            <div class="progress-bar"><div class="progress-fill" style="width:45%%; background: linear-gradient(90deg, #8b949e, #58a6ff);"></div></div>
        </div>
    </div>
    
    <div class="footer">
        <p>📅 Updated: {timestamp} &bull; Powered by Hybrid CNN-ViT (EfficientNet-B3 + DistilBERT) &bull; Thesis Project</p>
        <p style="margin-top:0.3rem;opacity:0.6;">🎯 75.33%% Test Accuracy &bull; 15.8x Improvement Over Random</p>
        <p style="margin-top:0.5rem;font-size:0.7rem;color:#8b949e;">💡 Note: Dashboard runs in demo mode. Full model inference is available on Kaggle.</p>
    </div>
</body>
</html>
'''

class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html; charset=utf-8')
        self.end_headers()
        html = HTML.replace('{timestamp}', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        self.wfile.write(html.encode('utf-8'))

print("🚀 Starting Dashboard (Demo Mode) at http://localhost:8080")
print("📊 Press Ctrl+C to stop")
webbrowser.open('http://localhost:8080')

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print(f"🌐 Open: http://localhost:{PORT}")
    httpd.serve_forever()