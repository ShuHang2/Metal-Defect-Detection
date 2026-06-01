import cv2
import numpy as np
import time
import yaml
import os
import glob
import json
import psutil
import subprocess
from flask import Flask, Response, request, jsonify
from threading import Thread
from ais_bench.infer.interface import InferSession
import webbrowser

app = Flask(__name__)

# ==================== 全局共享内存区 ====================
global_frame = None            
latest_clean_frame = None      
shared_boxes = []              
shared_scores = []             
shared_labels = []             
shared_fps = 0.0               

# 参数控制
CONF_THRESHOLD = 0.45
NMS_THRESHOLD = 0.45
INPUT_SIZE = 512
DEVICE_ID = 0

# 模型控制
DEFAULT_YAML = "GC10.yaml"
CURRENT_MODEL = "./GC10-512.om"
TARGET_MODEL = None   
INFER_ON = True       

# ROI 屏蔽 (变更为多边形列表，支持多个盲区)
shared_roi_polygons = []

# 配置保存文件
CONFIG_FILE = "saved_configs.json"
# =======================================================

class NPU_Detector:
    def __init__(self):
        self.class_names = {}
        self.load_yaml(CURRENT_MODEL) 

    def load_yaml(self, model_path):
        base_name = os.path.basename(model_path)
        
        yaml_path_direct = model_path.replace('.om', '.yaml')
        yaml_path_split = f"{base_name.split('-')[0]}.yaml" if '-' in base_name else yaml_path_direct
        
        if os.path.exists(yaml_path_direct):
            yaml_path = yaml_path_direct
        elif os.path.exists(yaml_path_split):
            yaml_path = yaml_path_split
        else:
            print(f"⚠️ 未找到对应的 YAML 标签文件，回退使用 {DEFAULT_YAML}")
            yaml_path = DEFAULT_YAML
            
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                names = data.get('names', {})
                if isinstance(names, list):
                    self.class_names = {i: str(n) for i, n in enumerate(names)}
                elif isinstance(names, dict):
                    self.class_names = {int(k): str(v) for k, v in names.items()}
            print(f"✅ 成功加载标签文件: {yaml_path}")
        except Exception as e:
            print(f"❌ 加载 YAML 失败: {e}")
            self.class_names = {}

    def preprocess(self, img):
        global shared_roi_polygons
        h, w = img.shape[:2]

        scale = max(h, w) / INPUT_SIZE
        new_h, new_w = int(h / scale), int(w / scale)
        resized = cv2.resize(img, (new_w, new_h))
        
        # 动态 ROI 盲区屏蔽 (多区域支持)
        if shared_roi_polygons and len(shared_roi_polygons) > 0:
            mask = np.ones((new_h, new_w), dtype=np.uint8) * 255
            for poly in shared_roi_polygons:
                if len(poly) >= 3:
                    pts = np.array([[int(p['x'] * new_w), int(p['y'] * new_h)] for p in poly], np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    cv2.fillPoly(mask, [pts], 0)
            resized = cv2.bitwise_and(resized, resized, mask=mask)

        canvas = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
        canvas[:new_h, :new_w] = resized
        blob = cv2.dnn.blobFromImage(canvas, scalefactor=1/255.0, swapRB=True)
        return blob, scale

    def postprocess(self, outputs, scale):
        global CONF_THRESHOLD, NMS_THRESHOLD 
        
        output = np.squeeze(outputs[0])
        if output.shape[0] < output.shape[1]: 
            output = output.T 
        
        scores = np.max(output[:, 4:], axis=1)
        mask = scores > CONF_THRESHOLD
        output, scores = output[mask], scores[mask]
        
        if len(scores) == 0: return [], [], []
        
        class_ids = np.argmax(output[:, 4:], axis=1)
        boxes = output[:, :4] * scale
        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2
        final_boxes = np.stack([x1, y1, x2, y2], axis=1).tolist()

        indices = cv2.dnn.NMSBoxes(final_boxes, scores.tolist(), CONF_THRESHOLD, NMS_THRESHOLD)
        if len(indices) == 0: return [], [], []
        
        idx = indices.flatten()
        return [final_boxes[i] for i in idx], scores[idx].tolist(), class_ids[idx].tolist()

    def camera_worker(self):
        global latest_clean_frame, global_frame, shared_fps
        global shared_boxes, shared_scores, shared_labels
        
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)
        
        while True:
            ret, frame = cap.read()
            if not ret: continue
            
            latest_clean_frame = frame.copy()
            
            # 实时渲染检测框
            if len(self.class_names) > 0 and len(shared_boxes) > 0:
                for box, score, cls_id in zip(shared_boxes, shared_scores, shared_labels):
                    cls_id = int(cls_id)
                    name = self.class_names.get(cls_id, f"Unknown_{cls_id}")
                    
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) 
                    cv2.putText(frame, f"{name} {score:.2f}", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            status_text = f"NPU FPS: {shared_fps:.1f}" if INFER_ON else "NPU STATUS: PAUSED"
            color = (0, 255, 0) if INFER_ON else (0, 0, 255)
            cv2.putText(frame, status_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            global_frame = frame

    def infer_worker(self):
        global latest_clean_frame, shared_boxes, shared_scores, shared_labels, shared_fps
        global CURRENT_MODEL, TARGET_MODEL, INFER_ON
        
        try:
            self.session = InferSession(device_id=DEVICE_ID, model_path=CURRENT_MODEL)
        except Exception as e:
            print(f"初始化模型失败: {e}")
            
        last_processed_id = 0
        prev_time = time.time()

        while True:
            if TARGET_MODEL is not None:
                shared_boxes, shared_scores, shared_labels = [], [], [] 
                try:
                    if hasattr(self, 'session'):
                        del self.session
                    self.session = InferSession(device_id=DEVICE_ID, model_path=TARGET_MODEL)
                    self.load_yaml(TARGET_MODEL)
                    CURRENT_MODEL = TARGET_MODEL
                except Exception as e:
                    try:
                        self.session = InferSession(device_id=DEVICE_ID, model_path=CURRENT_MODEL)
                        self.load_yaml(CURRENT_MODEL)
                    except: pass
                TARGET_MODEL = None 
                prev_time = time.time() 

            if not INFER_ON:
                shared_boxes, shared_scores, shared_labels = [], [], []
                shared_fps = 0.0
                time.sleep(0.1)
                continue

            frame = latest_clean_frame
            if frame is None or id(frame) == last_processed_id:
                time.sleep(0.005)
                continue
            
            last_processed_id = id(frame)
            blob, scale = self.preprocess(frame)
            
            try:
                outputs = self.session.infer(feeds=[np.ascontiguousarray(blob)], mode="static")
                shared_boxes, shared_scores, shared_labels = self.postprocess(outputs, scale)
            except:
                pass
            
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time + 1e-6)
            prev_time = curr_time
            shared_fps = shared_fps * 0.9 + fps * 0.1

# ==================== 配置存储管理 ====================
def load_configs_from_file():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except: pass
    return {}

def save_configs_to_file(configs):
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(configs, f, ensure_ascii=False, indent=4)

# ==================== Flask 路由与 API ====================
@app.route('/api/get_models', methods=['GET'])
def get_models():
    om_files = glob.glob("*.om")
    return jsonify({"models": om_files, "current": CURRENT_MODEL})

@app.route('/api/switch_model', methods=['POST'])
def switch_model():
    global TARGET_MODEL
    data = request.get_json()
    new_model = data.get('model_name')
    if new_model and os.path.exists(new_model):
        TARGET_MODEL = new_model
        return jsonify({"status": "success"})
    return jsonify({"status": "error"}), 400

@app.route('/api/toggle_infer', methods=['POST'])
def toggle_infer():
    global INFER_ON
    data = request.get_json()
    INFER_ON = data.get('state', True)
    return jsonify({"status": "success"})

@app.route('/api/update_params', methods=['POST'])
def update_params():
    global CONF_THRESHOLD, NMS_THRESHOLD
    data = request.get_json()
    CONF_THRESHOLD = float(data.get('conf', CONF_THRESHOLD))
    NMS_THRESHOLD = float(data.get('nms', NMS_THRESHOLD))
    return jsonify({"status": "success"})

@app.route('/api/set_roi', methods=['POST'])
def set_roi():
    global shared_roi_polygons
    data = request.get_json()
    polygons = data.get('polygons', [])
    shared_roi_polygons = polygons
    return jsonify({"status": "success"})

@app.route('/api/sys_info', methods=['GET'])
def sys_info():
    info = {
        "cpu": f"{psutil.cpu_percent(interval=None):.1f}%",
        "ram": f"{psutil.virtual_memory().percent:.1f}%",
        "npu": "N/A",
        "temp": "N/A"
    }
    try:
        npu_out = subprocess.check_output(["npu-smi", "info"], text=True, timeout=2)
        lines = npu_out.strip().split('\n')
        data_lines = [line for line in lines if line.startswith('| 0 ')]
        
        if len(data_lines) >= 2:
            row1_parts = data_lines[0].split('|')[3].strip().split()
            if len(row1_parts) >= 2:
                info["temp"] = f"{row1_parts[1]}°C"
            row2_parts = data_lines[1].split('|')[3].strip().split()
            if len(row2_parts) >= 1:
                info["npu"] = f"{row2_parts[0]}%"
    except Exception as e:
        pass
    return jsonify(info)

@app.route('/api/get_configs', methods=['GET'])
def get_configs():
    return jsonify(load_configs_from_file())

@app.route('/api/save_config', methods=['POST'])
def save_config():
    data = request.get_json()
    name = data.get('name', f"配置_{int(time.time())}")
    configs = load_configs_from_file()
    configs[name] = {
        "model": data.get("model", CURRENT_MODEL),
        "conf": float(data.get("conf", CONF_THRESHOLD)),
        "nms": float(data.get("nms", NMS_THRESHOLD)),
        "roi": data.get("roi", []) 
    }
    save_configs_to_file(configs)
    return jsonify({"status": "success"})

@app.route('/api/load_config', methods=['POST'])
def load_config():
    global TARGET_MODEL, CONF_THRESHOLD, NMS_THRESHOLD, shared_roi_polygons
    data = request.get_json()
    name = data.get('name')
    configs = load_configs_from_file()
    
    if name in configs:
        c = configs[name]
        if c.get("model") and os.path.exists(c.get("model")):
            TARGET_MODEL = c["model"]
            
        CONF_THRESHOLD = c.get("conf", CONF_THRESHOLD)
        NMS_THRESHOLD = c.get("nms", NMS_THRESHOLD)
        shared_roi_polygons = c.get("roi", [])
        
        return jsonify({
            "status": "success",
            "config": {
                "model": c.get("model", CURRENT_MODEL),
                "conf": CONF_THRESHOLD,
                "nms": NMS_THRESHOLD,
                "roi": shared_roi_polygons
            }
        })
    return jsonify({"status": "error"}), 400

@app.route('/api/delete_config', methods=['POST'])
def delete_config():
    data = request.get_json()
    name = data.get('name')
    configs = load_configs_from_file()
    if name in configs:
        del configs[name]
        save_configs_to_file(configs)
        return jsonify({"status": "success"})
    return jsonify({"status": "error"}), 400

def generate_video_stream():
    global global_frame
    last_sent_id = 0
    while True:
        if global_frame is None or id(global_frame) == last_sent_id:
            time.sleep(0.01)
            continue
        last_sent_id = id(global_frame)
        try:
            frame_to_send = cv2.resize(global_frame, (1920, 1080))
        except: continue
        ret, buffer = cv2.imencode('.jpg', frame_to_send, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    return """
    <html>
        <head>
            <title>边缘智能 NPU 工业质检台</title>
            <style>
                body { background:#121212; color:#e0e0e0; font-family:'Segoe UI', sans-serif; margin: 0; padding: 15px; }
                h2 { text-align: center; color: #ffffff; margin: 10px 0 20px 0; letter-spacing: 1px; font-size: 1.5em;}
                .main-layout { display: flex; flex-direction: row; justify-content: center; align-items: flex-start; gap: 20px; max-width: 1600px; margin: 0 auto; }
                .video-section { display: flex; flex-direction: column; gap: 15px; flex: 1; max-width: 1000px; }
                .top-bar { background: #232323; padding: 12px 20px; border-radius: 10px; border: 1px solid #333; display: flex; align-items: center; justify-content: space-between; gap: 20px; }
                .video-container { position: relative; width: 100%; border: 2px solid #444; border-radius: 8px; box-shadow: 0 8px 16px rgba(0,0,0,0.6); overflow: hidden; background: #000; }
                .video-container img { width: 100%; height: auto; display: block; }
                #roiCanvas { position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; }
                .status-bar { display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; }
                .status-card { background: #1a1a1a; padding: 12px; border-radius: 8px; border: 1px solid #333; text-align: center; }
                .status-label { font-size: 0.75em; color: #888; margin-bottom: 5px; text-transform: uppercase; }
                .status-value { font-size: 1.1em; font-weight: bold; color: #4CAF50; }
                .control-sidebar { width: 340px; display: flex; flex-direction: column; gap: 15px; flex-shrink: 0; }
                .controls { background: #232323; padding: 15px 20px; border-radius: 10px; border: 1px solid #333; }
                h3 { border-bottom: 1px solid #444; padding-bottom: 8px; margin-top: 0; color: #4CAF50; font-size: 1em; display: flex; align-items: center; gap: 8px; }
                .control-group, .slider-group { margin: 12px 0; display: flex; justify-content: space-between; align-items: center; }
                label { font-size: 0.85em; font-weight: 500; }
                input[type=range] { width: 140px; accent-color: #4CAF50; }
                select, input[type=text] { padding: 8px; border-radius: 6px; background: #111; color: white; border: 1px solid #444; outline: none; font-size: 0.9em; }
                span.val { font-weight: bold; color: #4CAF50; min-width: 35px; text-align: right; font-family: monospace; }
                .btn-row { display: flex; justify-content: space-between; gap: 8px; margin-top: 10px; }
                .btn { padding: 8px 10px; border: none; border-radius: 5px; font-weight: bold; cursor: pointer; transition: 0.2s; font-size: 0.8em; flex: 1; }
                .btn-draw { background: #2196F3; color: white; }
                .btn-apply { background: #4CAF50; color: white; }
                .btn-clear { background: #F44336; color: white; }
                .btn:hover { opacity: 0.8; }
                .switch { position: relative; display: inline-block; width: 40px; height: 20px; }
                .switch input { opacity: 0; width: 0; height: 0; }
                .slider { position: absolute; cursor: pointer; top: 0; left: 0; right: 0; bottom: 0; background-color: #555; transition: .4s; border-radius: 20px;}
                .slider:before { position: absolute; content: ""; height: 14px; width: 14px; left: 3px; bottom: 3px; background-color: white; transition: .4s; border-radius: 50%;}
                input:checked + .slider { background-color: #4CAF50; }
                input:checked + .slider:before { transform: translateX(20px); }
            </style>
        </head>
        <body>
            <h2>🏭 边缘智能 NPU 工业质检系统</h2>
            
            <div class="main-layout">
                <div class="video-section">
                    
                    <div class="top-bar">
                        <div style="display:flex; align-items:center; gap:15px;">
                            <label style="font-weight:bold; color:#4CAF50;">⚙️ 系统与模型</label>
                            <div class="switch-container" style="display:flex; align-items:center; gap:8px;">
                                <span style="font-size:0.8em;">检测开关</span>
                                <label class="switch"><input type="checkbox" id="inferToggle" checked onchange="toggleInfer()"><span class="slider"></span></label>
                            </div>
                        </div>
                        <div style="display:flex; align-items:center; gap:10px; flex:1; max-width:400px;">
                            <select id="modelSelect" onchange="switchModel()" style="width:100%;"></select>
                            <span id="switchStatus" style="color:#ffeb3b; font-size:0.7em; white-space:nowrap; width:60px;"></span>
                        </div>
                    </div>

                    <div class="video-container" id="vidContainer">
                        <img id="videoStream" src="/video_feed">
                        <canvas id="roiCanvas"></canvas>
                    </div>

                    <div class="status-bar">
                        <div class="status-card">
                            <div class="status-label">CPU Usage</div>
                            <div id="sysCpu" class="status-value">--%</div>
                        </div>
                        <div class="status-card">
                            <div class="status-label">Memory</div>
                            <div id="sysRam" class="status-value">--%</div>
                        </div>
                        <div class="status-card">
                            <div class="status-label">NPU Load</div>
                            <div id="sysNpu" class="status-value" style="color: #2196F3;">--%</div>
                        </div>
                        <div class="status-card">
                            <div class="status-label">NPU Temp</div>
                            <div id="sysTemp" class="status-value" style="color: #FF9800;">--°C</div>
                        </div>
                    </div>
                </div>

                <div class="control-sidebar">
                    
                    <div class="controls">
                        <h3>🎛️ 算法阈值</h3>
                        <div class="slider-group">
                            <label>置信度 (Conf)</label>
                            <div style="display:flex; align-items:center; gap:8px;">
                                <input type="range" id="confRange" min="0.05" max="0.95" step="0.05" value="0.45" oninput="updateParams()">
                                <span class="val" id="confVal">0.45</span>
                            </div>
                        </div>
                        <div class="slider-group">
                            <label>重叠率 (NMS)</label>
                            <div style="display:flex; align-items:center; gap:8px;">
                                <input type="range" id="nmsRange" min="0.05" max="0.95" step="0.05" value="0.45" oninput="updateParams()">
                                <span class="val" id="nmsVal">0.45</span>
                            </div>
                        </div>
                    </div>

                    <div class="controls">
                        <h3>💾 预设配置管理</h3>
                        <div style="display:flex; flex-direction:column; gap:10px;">
                            <input type="text" id="configName" placeholder="输入配置名称...">
                            <div style="display:flex; gap:8px;">
                                <select id="configSelect" style="flex:1;"><option value="">-- 选择预设 --</option></select>
                                <button class="btn btn-apply" onclick="saveConfig()" style="flex:0.4;">保存</button>
                            </div>
                            <div class="btn-row">
                                <button class="btn btn-draw" onclick="loadConfig()">加载配置</button>
                                <button class="btn btn-clear" onclick="deleteCurrentConfig()">删除选中</button>
                            </div>
                        </div>
                    </div>

                    <div class="controls">
                        <h3>🛡️ ROI 屏蔽盲区</h3>
                        <p style="font-size:0.7em; color:#888; margin:0 0 10px 0;">可多次点击绘制叠加多个区域。双击空白结束绘制。</p>
                        <div class="btn-row">
                            <button class="btn btn-draw" onclick="startDrawing()" id="drawBtn">🖌️ 开始绘制</button>
                            <button class="btn btn-apply" onclick="applyROI()">✅ 应用生效</button>
                            <button class="btn btn-clear" onclick="clearROI()">🗑️ 清除所有</button>
                        </div>
                    </div>

                </div>
            </div>

            <script>
                const canvas = document.getElementById('roiCanvas');
                const ctx = canvas.getContext('2d');
                const videoStream = document.getElementById('videoStream');
                
                let isDrawing = false;
                let allPolygons = []; // 存储所有已确认的多边形
                let points = [];      // 存储当前正在绘制的多边形点

                function syncCanvasSize() {
                    canvas.width = videoStream.clientWidth;
                    canvas.height = videoStream.clientHeight;
                    redraw();
                }
                window.addEventListener('resize', syncCanvasSize);
                videoStream.onload = syncCanvasSize;

                function startDrawing() {
                    isDrawing = true; 
                    points = []; // 仅清空当前正在画的线，不清理历史
                    canvas.style.pointerEvents = 'auto'; 
                    canvas.style.cursor = 'crosshair';
                    document.getElementById('drawBtn').innerText = '🔴 绘图中';
                    redraw();
                }

                canvas.addEventListener('mousedown', function(e) {
                    if (!isDrawing) return;
                    const rect = canvas.getBoundingClientRect();
                    points.push({
                        x: (e.clientX - rect.left) / rect.width,
                        y: (e.clientY - rect.top) / rect.height
                    });
                    redraw();
                });

                // 通用的绘制单个多边形函数
                function drawPoly(polyArray, isClosed) {
                    ctx.beginPath();
                    ctx.fillStyle = 'rgba(244, 67, 54, 0.3)'; 
                    ctx.strokeStyle = '#F44336';
                    ctx.lineWidth = 2;
                    polyArray.forEach((p, index) => {
                        const px = p.x * canvas.width, py = p.y * canvas.height;
                        if (index === 0) ctx.moveTo(px, py);
                        else ctx.lineTo(px, py);
                        ctx.fillRect(px - 3, py - 3, 6, 6);
                    });
                    if (isClosed && polyArray.length >= 3) {
                        ctx.closePath(); 
                        ctx.fill();
                    }
                    ctx.stroke();
                }

                function redraw() {
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                    
                    // 1. 优先绘制所有历史生效的多边形
                    allPolygons.forEach(poly => {
                        drawPoly(poly, true);
                    });

                    // 2. 绘制当前正在勾勒的多边形
                    if (points.length > 0) {
                        drawPoly(points, !isDrawing && points.length >= 3);
                    }
                }

                function applyROI() {
                    // 如果正在画的形状不足3个点，拦截
                    if (points.length > 0 && points.length < 3) { 
                        alert("当前多边形需至少 3 个点才能闭合！"); 
                        return; 
                    }
                    
                    // 将当前合法的多边形推入历史库
                    if (points.length >= 3) {
                        allPolygons.push([...points]);
                    }

                    isDrawing = false;
                    points = []; // 重置当前画笔
                    canvas.style.pointerEvents = 'none'; 
                    canvas.style.cursor = 'default';
                    document.getElementById('drawBtn').innerText = '🖌️ 开始绘制';
                    
                    redraw(); 
                    // 发送整个多边形列表给后台
                    fetch('/api/set_roi', { 
                        method: 'POST', 
                        headers: {'Content-Type': 'application/json'}, 
                        body: JSON.stringify({polygons: allPolygons}) 
                    });
                }

                function clearROI() {
                    isDrawing = false; 
                    points = []; 
                    allPolygons = []; // 清空历史所有多边形
                    canvas.style.pointerEvents = 'none';
                    document.getElementById('drawBtn').innerText = '🖌️ 开始绘制';
                    redraw();
                    fetch('/api/set_roi', { 
                        method: 'POST', 
                        headers: {'Content-Type': 'application/json'}, 
                        body: JSON.stringify({polygons: []}) 
                    });
                }

                function refreshConfigList() {
                    fetch('/api/get_configs').then(r => r.json()).then(data => {
                        const select = document.getElementById('configSelect');
                        select.innerHTML = '<option value="">-- 选择预设 --</option>';
                        for(let key in data) {
                            let option = document.createElement("option");
                            option.value = key; option.text = key;
                            select.add(option);
                        }
                    });
                }

                function saveConfig() {
                    let name = document.getElementById('configName').value.trim();
                    let selectedConfig = document.getElementById('configSelect').value;
                    if (!name) {
                        if (selectedConfig) name = selectedConfig;
                        else name = "配置_" + new Date().toLocaleTimeString();
                    }
                    const payload = {
                        name: name,
                        conf: document.getElementById('confRange').value,
                        nms: document.getElementById('nmsRange').value,
                        model: document.getElementById('modelSelect').value,
                        roi: allPolygons // 保存整个多边形列表
                    };
                    fetch('/api/save_config', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload) })
                    .then(res => res.json()).then(data => {
                        if(data.status === "success") {
                            alert("配置已保存");
                            document.getElementById('configName').value = "";
                            refreshConfigList();
                        }
                    });
                }

                function loadConfig() {
                    const selectedConfig = document.getElementById('configSelect').value;
                    if (!selectedConfig) return;
                    fetch('/api/load_config', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({name: selectedConfig}) })
                    .then(res => res.json()).then(data => {
                        if(data.status === "success") {
                            const c = data.config;
                            document.getElementById('confRange').value = c.conf;
                            document.getElementById('confVal').innerText = c.conf;
                            document.getElementById('nmsRange').value = c.nms;
                            document.getElementById('nmsVal').innerText = c.nms;
                            document.getElementById('modelSelect').value = c.model;
                            
                            allPolygons = c.roi ? c.roi : [];
                            points = []; 
                            redraw();
                        }
                    });
                }

                function deleteCurrentConfig() {
                    const selectedConfig = document.getElementById('configSelect').value;
                    if (!selectedConfig) return;
                    if (confirm("确定删除？")) {
                        fetch('/api/delete_config', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({name: selectedConfig}) })
                        .then(() => refreshConfigList());
                    }
                }

                window.onload = function() {
                    syncCanvasSize();
                    refreshConfigList(); 
                    fetch('/api/get_models').then(r => r.json()).then(data => {
                        const select = document.getElementById('modelSelect');
                        data.models.forEach(model => {
                            let option = document.createElement("option");
                            option.text = model; option.value = model;
                            if (data.current.includes(model)) option.selected = true;
                            select.add(option);
                        });
                    });
                };

                function switchModel() {
                    const selectedModel = document.getElementById('modelSelect').value;
                    const statusText = document.getElementById('switchStatus');
                    statusText.innerText = "🔄重载中";
                    fetch('/api/switch_model', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({model_name: selectedModel}) })
                    .then(res => res.json()).then(data => {
                        statusText.innerText = data.status === "success" ? "✅已就绪" : "❌失败";
                        setTimeout(() => statusText.innerText = "", 2000);
                    });
                }

                function toggleInfer() {
                    const isChecked = document.getElementById('inferToggle').checked;
                    fetch('/api/toggle_infer', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({state: isChecked}) });
                }

                function updateParams() {
                    const conf = document.getElementById('confRange').value;
                    const nms = document.getElementById('nmsRange').value;
                    document.getElementById('confVal').innerText = conf;
                    document.getElementById('nmsVal').innerText = nms;
                    fetch('/api/update_params', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({conf: conf, nms: nms}) });
                }

                setInterval(() => {
                    fetch('/api/sys_info').then(res => res.json()).then(data => {
                        document.getElementById('sysCpu').innerText = data.cpu;
                        document.getElementById('sysRam').innerText = data.ram;
                        document.getElementById('sysNpu').innerText = data.npu;
                        document.getElementById('sysTemp').innerText = data.temp;
                    }).catch(e => {});
                }, 2000);
            </script>
        </body>
    </html>
    """

@app.route('/video_feed')
def video_feed():
    return Response(generate_video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    detector = NPU_Detector()
    Thread(target=detector.camera_worker, daemon=True).start()
    Thread(target=detector.infer_worker, daemon=True).start()

    print("\n" + "="*60)
    print("🚀 边缘智能 NPU 工业质检台启动完毕！")
    print("👉 请在浏览器中访问: http://<香橙派IP地址>:5000")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)