# XiaoZhi ESP32 Server - Complete Deployment Guide

> **Mục đích**: Hướng dẫn chi tiết triển khai server XiaoZhi từ source code cho ESP32-C3 robot
> 
> **Cấu hình hệ thống**: Ubuntu PC - 6 cores, 16GB RAM, GTX 1060 6GB
> 
> **Chiến lược**: ASR & TTS local, LLM qua Gemini API

---

## 📋 MỤC LỤC

1. [Cài đặt từ Source](#1-cài-đặt-từ-source)
2. [Kiến trúc Source Code](#2-kiến-trúc-source-code)
3. [Configuration & Customization](#3-configuration--customization)
4. [ESP32-C3 Configuration](#4-esp32-c3-configuration)
5. [Troubleshooting & Performance](#5-troubleshooting--performance)

---

## 1. CÀI ĐẶT TỪ SOURCE

### 1.1. Chuẩn bị môi trường

#### 1.1.1. Kiểm tra CUDA (cho GPU acceleration)

```bash
# Kiểm tra CUDA đã cài chưa
nvidia-smi

# Kiểm tra CUDA version
nvcc --version

# Nếu chưa có CUDA, cài đặt CUDA 11.8 hoặc 12.1
# Ubuntu 22.04:
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-11-8

# Thêm vào ~/.bashrc
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

#### 1.1.2. Cài đặt Conda

```bash
# Download Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# Cài đặt
bash Miniconda3-latest-Linux-x86_64.sh

# Reload shell
source ~/.bashrc

# Verify
conda --version
```

#### 1.1.3. Tạo môi trường Python

```bash
# Tạo môi trường mới
conda create -n xiaozhi python=3.10 -y
conda activate xiaozhi

# Cài các dependencies hệ thống quan trọng
conda install libopus ffmpeg -y

# Linux specific: Cài thêm libiconv nếu cần
conda install libiconv -y
```

### 1.2. Clone source code

```bash
# Clone repository
cd ~/Desktop/RD
git clone https://github.com/xinnan-tech/xiaozhi-esp32-server.git
cd xiaozhi-esp32-server/main/xiaozhi-server

# Kiểm tra cấu trúc
ls -la
```

### 1.3. Cài đặt Python dependencies

```bash
# Đảm bảo đang trong môi trường conda
conda activate xiaozhi

# Cài PyTorch với CUDA support
# Cho GTX 1060 (CUDA 11.8 compatible)
pip install torch==2.2.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118

# Verify PyTorch GPU
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"

# Kết quả mong đợi:
# CUDA available: True
# CUDA version: 11.8
# Device: NVIDIA GeForce GTX 1060 6GB

# Cài các dependencies còn lại
pip install -r requirements.txt
```

### 1.4. Tải ASR model (FunASR)

```bash
# Tạo thư mục models
mkdir -p models/SenseVoiceSmall

# Tải model SenseVoiceSmall (~400MB)
# Option 1: Từ ModelScope (recommended)
cd models/SenseVoiceSmall
wget https://modelscope.cn/models/iic/SenseVoiceSmall/resolve/master/model.pt

# Option 2: Nếu link trên chậm, dùng mirror
wget https://hf-mirror.com/FunAudioLLM/SenseVoiceSmall/resolve/main/model.pt

# Verify
ls -lh model.pt
# Kết quả: ~395MB

cd ../..
```

### 1.5. Tạo cấu trúc thư mục

```bash
# Tạo các thư mục cần thiết
mkdir -p data tmp music

# Verify
tree -L 1
# Kết quả:
# .
# ├── app.py
# ├── config.yaml
# ├── data/
# ├── models/
# ├── tmp/
# ├── music/
# ├── core/
# └── ...
```

### 1.6. Tạo file cấu hình

```bash
# Tạo file .config.yaml trong thư mục data
touch data/.config.yaml
```

> **Lưu ý**: Nội dung file `.config.yaml` sẽ được đề cập chi tiết ở [Phần 3](#3-configuration--customization)

---

## 2. KIẾN TRÚC SOURCE CODE

### 2.1. Sơ đồ tổng quan

```
┌─────────────────────────────────────────────────────────────┐
│                        ESP32-C3 Robot                        │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │   MIC    │───▶│  AUDIO   │───▶│ WebSocket│              │
│  │          │    │ ENCODER  │    │  Client  │              │
│  └──────────┘    │ (OPUS)   │    └─────┬────┘              │
│                  └──────────┘          │                     │
└──────────────────────────────────────┼─────────────────────┘
                                        │ WS://ip:8000/xiaozhi/v1/
                                        ▼
┌─────────────────────────────────────────────────────────────┐
│              XiaoZhi Server (Ubuntu PC)                      │
│                                                               │
│  ┌────────────────────────────────────────────────────────┐ │
│  │              WebSocket Server (Port 8000)              │ │
│  │                  (core/websocket_server.py)            │ │
│  └────────────────┬───────────────────────────────────────┘ │
│                   │                                          │
│  ┌────────────────▼───────────────────────────────────────┐ │
│  │              Connection Handler                         │ │
│  │              (core/connection.py)                       │ │
│  └────────────────┬───────────────────────────────────────┘ │
│                   │                                          │
│  ┌────────────────▼───────────────────────────────────────┐ │
│  │     Audio Processing Pipeline (core/handle/)           │ │
│  │                                                         │ │
│  │  1. VAD (Voice Activity Detection)                     │ │
│  │     └─▶ SileroVAD (models/snakers4_silero-vad)        │ │
│  │                                                         │ │
│  │  2. ASR (Speech Recognition)                           │ │
│  │     └─▶ FunASR (models/SenseVoiceSmall) [GPU]         │ │
│  │                                                         │ │
│  │  3. Intent Recognition                                  │ │
│  │     └─▶ function_call (plugins_func/functions/)        │ │
│  │                                                         │ │
│  │  4. LLM (Language Model)                               │ │
│  │     └─▶ Gemini API (google-generativeai)              │ │
│  │                                                         │ │
│  │  5. TTS (Text-to-Speech)                               │ │
│  │     └─▶ EdgeTTS (Microsoft) [Streaming]               │ │
│  │                                                         │ │
│  └────────────────┬───────────────────────────────────────┘ │
│                   │                                          │
│  ┌────────────────▼───────────────────────────────────────┐ │
│  │        Audio Response (OPUS encoded)                    │ │
│  │        Send back via WebSocket                          │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                               │
│  ┌─────────────────────────────────────────────────────────┐│
│  │        HTTP Server (Port 8003)                          ││
│  │        - OTA firmware updates                           ││
│  │        - Vision analysis API                            ││
│  │        (core/http_server.py)                            ││
│  └─────────────────────────────────────────────────────────┘│
└───────────────────────────────────────────────────────────────┘
```

### 2.2. Chi tiết các module chính

#### 2.2.1. Entry Point (`app.py`)

**Chức năng**:
- Khởi động asyncio event loop
- Load config từ `data/.config.yaml` hoặc `config.yaml`
- Khởi động WebSocket server và HTTP server song song
- Xử lý graceful shutdown

**Code flow**:
```python
async def main():
    check_ffmpeg_installed()           # Kiểm tra ffmpeg
    config = load_config()             # Load config
    
    # Khởi tạo servers
    ws_server = WebSocketServer(config)
    ota_server = SimpleHttpServer(config)
    
    # Start servers concurrently
    ws_task = asyncio.create_task(ws_server.start())
    ota_task = asyncio.create_task(ota_server.start())
    
    await wait_for_exit()              # Block until SIGTERM/Ctrl-C
```

#### 2.2.2. WebSocket Server (`core/websocket_server.py`)

**Chức năng**:
- Lắng nghe kết nối WebSocket từ ESP32
- Tạo `Connection` instance cho mỗi client
- Quản lý lifecycle của connections

**Key methods**:
```python
class WebSocketServer:
    async def start(self):
        # Start WebSocket server on port 8000
        async with websockets.serve(
            self.handler, 
            self.ip, 
            self.port
        ):
            await asyncio.Future()  # Run forever
    
    async def handler(self, websocket, path):
        # Create connection instance
        connection = Connection(websocket, self.config)
        await connection.handle()
```

#### 2.2.3. Connection Handler (`core/connection.py`)

**Chức năng**: Xử lý toàn bộ pipeline cho 1 kết nối

**Pipeline stages**:

1. **Receive Audio Stream**: Nhận OPUS audio frames từ ESP32
2. **VAD Processing**: Phát hiện khi nào người dùng bắt đầu/kết thúc nói
3. **ASR Processing**: Chuyển audio → text
4. **Intent Recognition**: Phân loại intent (music, weather, chat, exit...)
5. **LLM Processing**: Tạo response text
6. **TTS Processing**: Chuyển text → audio
7. **Send Response**: Gửi audio về ESP32

**Code structure**:
```python
class Connection:
    async def handle(self):
        while True:
            # Receive audio frame
            frame = await self.websocket.recv()
            
            # VAD: Check if speaking
            if self.vad_provider.is_speech(frame):
                self.audio_buffer.append(frame)
            else:
                # End of speech detected
                if self.audio_buffer:
                    await self.process_speech()
    
    async def process_speech(self):
        # 1. ASR
        text = await self.asr_provider.transcribe(audio_buffer)
        
        # 2. Intent
        intent = await self.intent_provider.recognize(text)
        
        # 3. LLM
        response = await self.llm_provider.chat(text, intent)
        
        # 4. TTS
        audio = await self.tts_provider.synthesize(response)
        
        # 5. Send back
        await self.websocket.send(audio)
```

#### 2.2.4. Providers (`core/providers/`)

**Cấu trúc**:
```
core/providers/
├── asr/              # Speech Recognition
│   ├── fun_asr.py    # FunASR local (GPU)
│   ├── doubao_asr.py
│   └── ...
├── tts/              # Text-to-Speech
│   ├── edge_tts.py   # EdgeTTS (free)
│   ├── doubao_tts.py
│   └── ...
├── llm/              # Language Models
│   ├── gemini_llm.py # Google Gemini
│   ├── chatglm_llm.py
│   └── ...
├── vad/              # Voice Activity Detection
│   └── silero_vad.py
├── intent/           # Intent Recognition
│   ├── function_call.py
│   └── intent_llm.py
└── memory/           # Conversation Memory
    ├── nomem.py
    └── mem0ai.py
```

**Provider pattern**: Mỗi provider implement interface chuẩn

```python
# Example: ASR Provider
class ASRProvider(ABC):
    @abstractmethod
    async def transcribe(self, audio_data: bytes) -> str:
        """Convert audio to text"""
        pass

# FunASR implementation
class FunASR(ASRProvider):
    def __init__(self, config):
        self.model = self.load_model(config['model_dir'])
        if torch.cuda.is_available():
            self.model = self.model.cuda()  # GPU acceleration
    
    async def transcribe(self, audio_data: bytes) -> str:
        # Decode OPUS → PCM
        pcm = decode_opus(audio_data)
        
        # Run inference on GPU
        with torch.no_grad():
            result = self.model(pcm)
        
        return result['text']
```

#### 2.2.5. Plugin System (`plugins_func/functions/`)

**Chức năng**: Mở rộng khả năng của robot qua function calling

**Available plugins**:
```
plugins_func/functions/
├── get_weather.py           # Lấy thông tin thời tiết
├── get_news_from_newsnow.py # Đọc tin tức
├── play_music.py            # Phát nhạc từ thư mục
├── change_role.py           # Đổi nhân cách
├── hass_get_state.py        # Home Assistant control
└── ...
```

**Plugin structure**:
```python
# Example: get_weather.py
async def get_weather(location: str = None) -> str:
    """
    Lấy thông tin thời tiết
    
    Args:
        location: Tên thành phố (nếu không có dùng default)
    
    Returns:
        Thông tin thời tiết dạng text
    """
    api_key = config['plugins']['get_weather']['api_key']
    url = f"https://api.qweather.com/v7/weather/now"
    
    response = await httpx.get(url, params={
        'location': location,
        'key': api_key
    })
    
    data = response.json()
    return format_weather(data)

# Function metadata for LLM
FUNCTION_SCHEMA = {
    "name": "get_weather",
    "description": "Lấy thông tin thời tiết hiện tại",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "Tên thành phố"
            }
        }
    }
}
```

### 2.3. Data Flow chi tiết

```
[ESP32 Audio Stream]
        │
        ▼
[WebSocket Receive] ──┐
        │             │
        ▼             │ Continuous streaming
[OPUS Decode]         │
        │             │
        ▼             │
[VAD Detection] ◀─────┘
        │
        ├─▶ [Silence] ──▶ Continue buffering
        │
        ├─▶ [Speech Start] ──▶ Start recording
        │
        └─▶ [Speech End] ──▶ Process pipeline
                              │
                              ▼
                        [Audio Buffer]
                              │
                              ▼
                        [ASR: FunASR]
                        - Decode audio
                        - GPU inference
                        - Return text
                              │
                              ▼
                        [Text: "今天天气怎么样"]
                              │
                              ▼
                        [Intent Recognition]
                        - Parse intent
                        - Check wake word
                        - Detect function call
                              │
                              ├─▶ [Exit] ──▶ Close connection
                              │
                              ├─▶ [Function Call]
                              │        │
                              │        ▼
                              │   [Execute Plugin]
                              │   (e.g. get_weather)
                              │        │
                              │        ▼
                              │   [Function Result]
                              │        │
                              │        └──▶ [Merge with context]
                              │
                              ▼
                        [LLM Processing]
                        - Send to Gemini API
                        - Get response
                              │
                              ▼
                        [Response Text: "今天广州...]
                              │
                              ▼
                        [TTS: EdgeTTS]
                        - Streaming synthesis
                        - Convert to OPUS
                              │
                              ▼
                        [Audio Frames]
                              │
                              ▼
                        [WebSocket Send]
                              │
                              ▼
                        [ESP32 Speaker]
```

### 2.4. Concurrent Processing

Server sử dụng Python `asyncio` để xử lý concurrent connections:

```python
# Multiple ESP32 devices can connect simultaneously
connections = []

async def handle_connection(websocket):
    connection = Connection(websocket, config)
    connections.append(connection)
    
    try:
        await connection.handle()  # Each runs independently
    finally:
        connections.remove(connection)

# Server can handle 10-50 concurrent connections
# Limited by GPU memory for ASR processing
```

**Resource sharing**:
- **VAD model**: Loaded once, shared by all connections (lightweight)
- **ASR model**: Loaded once, GPU inference serialized via semaphore
- **LLM**: API calls, naturally concurrent
- **TTS**: Streaming, concurrent processing

**GPU memory management**:
```python
# ASR processing với semaphore để tránh OOM
asr_semaphore = asyncio.Semaphore(2)  # Max 2 concurrent ASR

async def transcribe(audio):
    async with asr_semaphore:
        # Only 2 ASR operations at a time on GPU
        result = await self.asr_model(audio)
    return result
```

---

## 3. CONFIGURATION & CUSTOMIZATION

### 3.1. Config file structure

Server đọc config theo thứ tự ưu tiên:
1. `data/.config.yaml` (highest priority - your custom config)
2. `config.yaml` (fallback - default config)

**Best practice**: Chỉ đưa những config cần override vào `data/.config.yaml`

### 3.2. Optimal configuration cho GTX 1060

Tạo file `data/.config.yaml` với nội dung sau:

```yaml
# ============================================================
# XiaoZhi Server Configuration
# Optimized for: GTX 1060 6GB, 16GB RAM, 6 cores
# Strategy: Local ASR+TTS, Gemini LLM
# ============================================================

# --------------------- Server Settings ---------------------
server:
  ip: 0.0.0.0
  port: 8000
  http_port: 8003
  # Thay YOUR_LOCAL_IP bằng IP thực của PC (vd: 192.168.1.100)
  websocket: ws://YOUR_LOCAL_IP:8000/xiaozhi/v1/
  vision_explain: http://YOUR_LOCAL_IP:8003/mcp/vision/explain
  
  # Timezone offset for Vietnam
  timezone_offset: +7
  
  # Authentication (optional - disable for local testing)
  auth:
    enabled: false

# --------------------- Logging Settings ---------------------
log:
  log_level: INFO  # Change to DEBUG for troubleshooting
  log_dir: tmp
  log_file: "server.log"

# --------------------- Performance Tuning ---------------------
# Xóa audio sau khi dùng để tiết kiệm disk
delete_audio: true

# Timeout cho TTS (tăng lên nếu mạng chậm)
tts_timeout: 10

# Đóng connection sau 2 phút không có audio
close_connection_no_voice_time: 120

# Enable wakeup word caching để tăng tốc
enable_wakeup_words_response_cache: true

# TTS audio send delay (0 = auto, based on frame rate)
tts_audio_send_delay: 0

# --------------------- AI Personality ---------------------
prompt: |
  Bạn là trợ lý AI thông minh, thân thiện và hữu ích.
  Bạn trả lời ngắn gọn, súc tích, dễ hiểu.
  Bạn có thể nói tiếng Việt và tiếng Anh.
  Khi được hỏi về thời tiết, tin tức, bạn sử dụng tools để lấy thông tin.

# --------------------- Module Selection ---------------------
selected_module:
  VAD: SileroVAD          # Voice activity detection
  ASR: FunASR             # Local ASR with GPU
  LLM: GeminiLLM          # Google Gemini API
  TTS: EdgeTTS            # Microsoft Edge TTS (free, streaming)
  Memory: nomem           # No memory for faster response
  Intent: function_call   # Function calling for plugins

# --------------------- VAD Configuration ---------------------
VAD:
  SileroVAD:
    type: silero
    threshold: 0.5        # Speech detection threshold
    threshold_low: 0.3    # Lower threshold for continuation
    model_dir: models/snakers4_silero-vad
    min_silence_duration_ms: 200  # Tăng lên 300-400 nếu bị cắt giữa câu

# --------------------- ASR Configuration ---------------------
ASR:
  FunASR:
    type: fun_local
    model_dir: models/SenseVoiceSmall
    output_dir: tmp/
    # GPU will be auto-detected and used if available

# --------------------- LLM Configuration ---------------------
LLM:
  GeminiLLM:
    type: gemini
    # Get your API key from: https://aistudio.google.com/apikey
    api_key: YOUR_GEMINI_API_KEY_HERE
    model_name: "gemini-2.0-flash-exp"  # Fast & free model
    # Nếu không truy cập được từ VN, bật proxy:
    # http_proxy: "http://127.0.0.1:7890"
    # https_proxy: "http://127.0.0.1:7890"

# --------------------- TTS Configuration ---------------------
TTS:
  EdgeTTS:
    type: edge
    # Vietnamese voices:
    # - vi-VN-HoaiMyNeural (Female)
    # - vi-VN-NamMinhNeural (Male)
    voice: vi-VN-HoaiMyNeural
    output_dir: tmp/

# --------------------- Intent Recognition ---------------------
Intent:
  function_call:
    type: function_call
    # Enabled plugins (comment out unused ones)
    functions:
      - get_weather         # Weather information
      - get_news_from_newsnow  # News
      - play_music          # Music playback (if you have music in ./music/)
      # - change_role       # Change AI personality
      # - hass_get_state    # Home Assistant (if configured)

# --------------------- Plugins Configuration ---------------------
plugins:
  get_weather:
    # Free API key for testing (limited requests)
    # Register your own at: https://console.qweather.com/#/apps/create-key/over
    api_host: "mj7p3y7naa.re.qweatherapi.com"
    api_key: "a861d0d5e7bf4ee1a83d9a9e4f96d4da"
    default_location: "Ho Chi Minh"  # Your city
  
  get_news_from_newsnow:
    url: "https://newsnow.busiyi.world/api/s?id="
    news_sources: "VnExpress;Tuổi Trẻ;Thanh Niên"
  
  play_music:
    music_dir: "./music"
    music_ext:
      - ".mp3"
      - ".wav"
    refresh_time: 300

# --------------------- Wakeup Words ---------------------
wakeup_words:
  - "hey robot"
  - "xin chào"
  - "hello"

# --------------------- Exit Commands ---------------------
exit_commands:
  - "tạm biệt"
  - "goodbye"
  - "exit"
```

### 3.3. Alternative TTS Options

Nếu EdgeTTS có độ trễ cao, thử các options sau:

#### Option 1: LinkeraiTTS (Free, streaming, Chinese service)

```yaml
selected_module:
  TTS: LinkeraiTTS

TTS:
  LinkeraiTTS:
    type: linkerai
    api_url: https://tts.linkerai.cn/tts
    audio_format: "pcm"
    access_token: "U4YdYXVfpwWnk2t5Gp822zWPCuORyeJL"  # Free testing token
    voice: "OUeAo1mhq6IBExi"
    output_dir: tmp/
```

#### Option 2: Local TTS với Fish-Speech (Requires more GPU RAM)

```bash
# Cài đặt Fish-Speech server (requires ~4GB GPU RAM)
docker pull fishaudio/fish-speech:latest
docker run -d -p 8080:8080 --gpus all fishaudio/fish-speech:latest

# Config
```

```yaml
selected_module:
  TTS: FishSpeech

TTS:
  FishSpeech:
    type: fishspeech
    api_url: "http://127.0.0.1:8080/v1/tts"
    api_key: "your_key"
    output_dir: tmp/
    # Voice cloning: Upload reference audio
    reference_audio: ["config/assets/my_voice.wav"]
    reference_text: ["Đây là giọng nói mẫu của tôi"]
```

### 3.4. Advanced Customization

#### 3.4.1. Custom Plugin Development

Tạo plugin mới trong `plugins_func/functions/`:

```python
# plugins_func/functions/get_crypto_price.py

import httpx
from typing import Optional

async def get_crypto_price(symbol: str = "BTC") -> str:
    """
    Lấy giá cryptocurrency hiện tại
    
    Args:
        symbol: Mã coin (BTC, ETH, BNB...)
    
    Returns:
        Thông tin giá coin
    """
    url = f"https://api.binance.com/api/v3/ticker/price"
    
    async with httpx.AsyncClient() as client:
        response = await client.get(url, params={
            'symbol': f'{symbol.upper()}USDT'
        })
        
        if response.status_code == 200:
            data = response.json()
            price = float(data['price'])
            return f"{symbol} hiện đang {price:,.2f} USDT"
        else:
            return f"Không tìm thấy giá cho {symbol}"

# Function schema cho LLM function calling
FUNCTION_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_crypto_price",
        "description": "Lấy giá cryptocurrency hiện tại từ Binance",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Mã cryptocurrency (VD: BTC, ETH, BNB)",
                    "enum": ["BTC", "ETH", "BNB", "SOL", "ADA"]
                }
            },
            "required": []
        }
    }
}
```

Thêm vào config:

```yaml
Intent:
  function_call:
    functions:
      - get_weather
      - get_crypto_price  # Your new plugin
```

#### 3.4.2. Custom LLM Provider

Tạo provider cho local LLM (VD: Ollama):

```python
# core/providers/llm/ollama_llm.py

import httpx
from typing import AsyncIterator
from .base import LLMProvider

class OllamaLLM(LLMProvider):
    def __init__(self, config):
        self.base_url = config.get('base_url', 'http://localhost:11434')
        self.model = config.get('model_name', 'llama3.2:3b')
    
    async def chat(
        self, 
        messages: list,
        stream: bool = True
    ) -> AsyncIterator[str]:
        """
        Chat với Ollama local model
        """
        url = f"{self.base_url}/api/chat"
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream(
                'POST',
                url,
                json={
                    'model': self.model,
                    'messages': messages,
                    'stream': stream
                }
            ) as response:
                async for line in response.aiter_lines():
                    if line:
                        data = json.loads(line)
                        if 'message' in data:
                            yield data['message']['content']
```

Config:

```yaml
selected_module:
  LLM: OllamaLLM

LLM:
  OllamaLLM:
    type: ollama
    base_url: http://localhost:11434
    model_name: qwen2.5:3b  # Lightweight model for 6GB GPU
```

#### 3.4.3. Performance Tuning Tips

**1. GPU Memory Management**

```python
# Modify core/providers/asr/fun_asr.py

# Reduce batch size if OOM
self.batch_size = 1  # Process 1 audio at a time

# Enable FP16 inference
if torch.cuda.is_available():
    self.model = self.model.half()  # FP16 -> 2x faster, 2x less VRAM
```

**2. VAD Tuning**

```yaml
VAD:
  SileroVAD:
    threshold: 0.4              # Lower = more sensitive (detect softer voice)
    threshold_low: 0.2          # Lower boundary
    min_silence_duration_ms: 300  # Longer silence before cutting
```

**3. Connection Pooling**

```python
# Modify core/connection.py

# Reuse HTTP clients
from httpx import AsyncClient

class Connection:
    def __init__(self):
        self.http_client = AsyncClient(
            timeout=30.0,
            limits=httpx.Limits(
                max_connections=100,
                max_keepalive_connections=20
            )
        )
```

### 3.5. Environment Variables

Tạo file `.env` để quản lý secrets:

```bash
# .env
GEMINI_API_KEY=your_actual_api_key_here
WEATHER_API_KEY=your_weather_key
```

Load trong code:

```python
# config/config_loader.py
import os
from dotenv import load_dotenv

load_dotenv()

def load_config():
    config = load_yaml()
    
    # Override with env vars
    if 'LLM' in config and 'GeminiLLM' in config['LLM']:
        config['LLM']['GeminiLLM']['api_key'] = os.getenv(
            'GEMINI_API_KEY',
            config['LLM']['GeminiLLM']['api_key']
        )
    
    return config
```

---

## 4. ESP32-C3 CONFIGURATION

### 4.1. Phương án A: Sử dụng Firmware có sẵn (Recommended)

#### 4.1.1. Flash firmware từ XiaoZhi

```bash
# Download firmware từ releases
wget https://github.com/78/xiaozhi-esp32/releases/latest/download/xiaozhi-esp32-c3.bin

# Flash với esptool
pip install esptool

# Tìm serial port
ls /dev/ttyUSB*  # hoặc /dev/ttyACM*

# Flash
esptool.py --chip esp32c3 --port /dev/ttyUSB0 --baud 460800 \
  write_flash -z 0x0 xiaozhi-esp32-c3.bin

# Monitor
esptool.py --chip esp32c3 --port /dev/ttyUSB0 monitor
```

#### 4.1.2. Cấu hình qua Web Interface

1. **Kết nối WiFi**:
   - ESP32 sẽ tạo AP: `XiaoZhi-XXXXXX`
   - Kết nối vào AP này
   - Mở browser: `http://192.168.4.1`

2. **Config WiFi**:
   - Chọn WiFi network của bạn
   - Nhập password
   - Save & Reboot

3. **Config Server**:
   - Sau khi reboot, ESP32 sẽ kết nối WiFi nhà
   - Tìm IP của ESP32 qua router hoặc serial monitor
   - Truy cập: `http://<ESP32_IP>`
   - Vào tab "Server Settings"
   - Nhập WebSocket URL: `ws://YOUR_PC_IP:8000/xiaozhi/v1/`
   - Save

4. **Test**:
   - Nhấn nút Boot trên ESP32 để trigger wake word
   - Nói: "Hello"
   - Kiểm tra logs server

### 4.2. Phương án B: Tự build firmware (Advanced)

#### 4.2.1. Setup ESP-IDF

```bash
# Install dependencies
sudo apt-get install git wget flex bison gperf python3 python3-pip \
  python3-venv cmake ninja-build ccache libffi-dev libssl-dev \
  dfu-util libusb-1.0-0

# Clone ESP-IDF
mkdir -p ~/esp
cd ~/esp
git clone --recursive https://github.com/espressif/esp-idf.git
cd esp-idf
git checkout v5.1.2

# Install
./install.sh esp32c3

# Setup env
. ./export.sh
```

#### 4.2.2. Clone XiaoZhi ESP32 project

```bash
cd ~/esp
git clone https://github.com/78/xiaozhi-esp32.git
cd xiaozhi-esp32
```

#### 4.2.3. Configure

```bash
# Open menuconfig
idf.py menuconfig

# Navigate to "XiaoZhi Configuration"
# Set:
# - WiFi SSID: your_wifi_name
# - WiFi Password: your_password
# - Server URL: ws://YOUR_PC_IP:8000/xiaozhi/v1/

# Save and exit
```

#### 4.2.4. Build & Flash

```bash
# Build
idf.py build

# Flash
idf.py -p /dev/ttyUSB0 flash

# Monitor
idf.py -p /dev/ttyUSB0 monitor
```

### 4.3. Firmware Configuration Files

ESP32 firmware config được lưu trong flash:

```
/spiffs/config.json
{
  "wifi": {
    "ssid": "YourWiFi",
    "password": "password"
  },
  "server": {
    "url": "ws://192.168.1.100:8000/xiaozhi/v1/",
    "token": "optional_auth_token"
  },
  "audio": {
    "sample_rate": 16000,
    "channels": 1,
    "format": "opus"
  }
}
```

### 4.4. Network Considerations

#### 4.4.1. Local Network Setup

```
┌─────────────────┐
│   WiFi Router   │
│  192.168.1.1    │
└────────┬────────┘
         │
         ├─────────────┐
         │             │
    ┌────▼──────┐ ┌───▼─────────┐
    │  PC       │ │  ESP32-C3   │
    │192.168.1.x│ │ 192.168.1.y │
    └───────────┘ └─────────────┘
```

**Requirements**:
- PC và ESP32 cùng subnet
- Firewall cho phép port 8000, 8003
- Router không block WebSocket traffic

#### 4.4.2. Firewall Configuration

```bash
# Ubuntu: Allow ports
sudo ufw allow 8000/tcp
sudo ufw allow 8003/tcp
sudo ufw reload

# Check
sudo ufw status
```

#### 4.4.3. Test Connection từ ESP32

```bash
# Từ ESP32 serial monitor:
> ping 192.168.1.100  # Your PC IP

# Should see:
# 64 bytes from 192.168.1.100: icmp_seq=1 ttl=64 time=2 ms
```

### 4.5. OTA Updates

Server cung cấp OTA endpoint:

```
http://YOUR_PC_IP:8003/xiaozhi/ota/
```

**Cách update firmware OTA**:

1. Build firmware mới
2. Upload lên server:
```bash
cp build/xiaozhi-esp32.bin main/xiaozhi-server/data/firmware.bin
```

3. Trigger OTA từ ESP32:
   - Web UI: Settings → OTA Update
   - Hoặc REST API:
```bash
curl -X POST http://<ESP32_IP>/api/ota \
  -d '{"url": "http://YOUR_PC_IP:8003/xiaozhi/ota/firmware.bin"}'
```

---

## 5. TROUBLESHOOTING & PERFORMANCE

### 5.1. Common Issues

#### 5.1.1. CUDA Out of Memory

**Triệu chứng**:
```
RuntimeError: CUDA out of memory. Tried to allocate XX MiB
```

**Solutions**:

1. Giảm concurrent connections:
```python
# core/websocket_server.py
MAX_CONNECTIONS = 2  # Reduce from default
```

2. Enable FP16:
```python
# core/providers/asr/fun_asr.py
self.model = self.model.half()  # Use FP16
```

3. Clear cache thường xuyên:
```python
import torch
torch.cuda.empty_cache()
```

#### 5.1.2. High Latency

**Symptoms**: Phản hồi chậm > 3 giây

**Diagnosis**:
```python
# Add timing logs in core/connection.py
import time

start = time.time()
text = await self.asr_provider.transcribe(audio)
print(f"ASR took: {time.time() - start:.2f}s")

start = time.time()
response = await self.llm_provider.chat(text)
print(f"LLM took: {time.time() - start:.2f}s")

start = time.time()
audio = await self.tts_provider.synthesize(response)
print(f"TTS took: {time.time() - start:.2f}s")
```

**Bottlenecks & Solutions**:

1. **ASR slow (>1s)**:
   - Check GPU utilization: `nvidia-smi`
   - Ensure CUDA is used: `torch.cuda.is_available()`
   - Reduce audio buffer size

2. **LLM slow (>2s)**:
   - Check internet connection
   - Use faster model: `gemini-2.0-flash-exp`
   - Enable streaming response
   - Consider local LLM (Ollama)

3. **TTS slow (>1s)**:
   - Switch to streaming TTS (EdgeTTS, LinkeraiTTS)
   - Reduce response length (shorter prompts)

#### 5.1.3. Connection Drops

**Symptoms**: WebSocket disconnects frequently

**Causes & Solutions**:

1. **Network instability**:
```yaml
# Increase timeout in config
close_connection_no_voice_time: 300  # 5 minutes
```

2. **ESP32 WiFi issues**:
```c
// In ESP32 firmware: Increase WiFi power
esp_wifi_set_ps(WIFI_PS_NONE);  // Disable power saving
```

3. **Server overload**:
```python
# Monitor server load
import psutil
print(f"CPU: {psutil.cpu_percent()}%")
print(f"RAM: {psutil.virtual_memory().percent}%")
```

#### 5.1.4. Audio Quality Issues

**Problem**: Tiếng bị méo, nhiễu

**Solutions**:

1. **Check OPUS encoding**:
```yaml
xiaozhi:
  audio_params:
    format: opus
    sample_rate: 16000  # Don't change
    channels: 1         # Mono
    frame_duration: 60  # milliseconds
```

2. **Adjust VAD sensitivity**:
```yaml
VAD:
  SileroVAD:
    threshold: 0.5  # Higher = less sensitive, less noise
```

3. **Check network packet loss**:
```bash
# From server, ping ESP32
ping -c 100 <ESP32_IP>

# Check packet loss percentage
# Should be <1%
```

### 5.2. Performance Benchmarks

Expected latency on GTX 1060 setup:

| Component | Time | Notes |
|-----------|------|-------|
| VAD | <50ms | Per frame, CPU |
| ASR (FunASR GPU) | 200-500ms | Depends on audio length |
| Intent Recognition | <100ms | Function calling |
| LLM (Gemini) | 500-1500ms | Network dependent |
| TTS (EdgeTTS) | 300-800ms | Streaming, first chunk |
| **Total (avg)** | **1.5-3s** | From speech end to audio start |

**Optimization targets**:
- End-to-end latency: <2s
- GPU utilization: 30-60%
- RAM usage: <4GB
- CPU usage: <40%

### 5.3. Monitoring & Logging

#### 5.3.1. Enable Debug Logging

```yaml
log:
  log_level: DEBUG  # More detailed logs
```

#### 5.3.2. Performance Monitoring Script

Tạo `monitor.py`:

```python
#!/usr/bin/env python3
import psutil
import time
from rich.console import Console
from rich.table import Table

console = Console()

def monitor():
    while True:
        # GPU stats
        import subprocess
        gpu_info = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', 
             '--format=csv,noheader,nounits']
        ).decode().strip().split(',')
        
        gpu_util = gpu_info[0]
        gpu_mem = gpu_info[1]
        
        # System stats
        cpu = psutil.cpu_percent(interval=1)
        mem = psutil.virtual_memory()
        
        # Display
        table = Table(title="XiaoZhi Server Monitor")
        table.add_column("Metric")
        table.add_column("Value")
        
        table.add_row("CPU", f"{cpu}%")
        table.add_row("RAM", f"{mem.percent}% ({mem.used/1e9:.1f}GB/{mem.total/1e9:.1f}GB)")
        table.add_row("GPU Util", f"{gpu_util}%")
        table.add_row("GPU Mem", f"{gpu_mem}MB")
        
        console.clear()
        console.print(table)
        
        time.sleep(2)

if __name__ == '__main__':
    monitor()
```

Run:
```bash
python monitor.py
```

#### 5.3.3. Log Analysis

```bash
# View real-time logs
tail -f tmp/server.log

# Filter errors
grep ERROR tmp/server.log

# Count requests per minute
grep "ASR transcribe" tmp/server.log | awk '{print $1}' | uniq -c

# Average response time (if logged)
grep "Total latency" tmp/server.log | awk '{sum+=$NF; count++} END {print sum/count}'
```

### 5.4. Production Deployment Tips

#### 5.4.1. Use Process Manager

```bash
# Install supervisor
sudo apt-get install supervisor

# Create supervisor config
sudo nano /etc/supervisor/conf.d/xiaozhi.conf
```

```ini
[program:xiaozhi]
directory=/home/misa/Desktop/RD/xiaozhi-esp32-server/main/xiaozhi-server
command=/home/misa/miniconda3/envs/xiaozhi/bin/python app.py
user=misa
autostart=true
autorestart=true
stderr_logfile=/var/log/xiaozhi.err.log
stdout_logfile=/var/log/xiaozhi.out.log
```

```bash
# Start
sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl start xiaozhi

# Check status
sudo supervisorctl status
```

#### 5.4.2. Auto-start on Boot

```bash
# Enable supervisor
sudo systemctl enable supervisor

# Verify
sudo systemctl status supervisor
```

#### 5.4.3. Backup Strategy

```bash
# Backup script
#!/bin/bash
DATE=$(date +%Y%m%d)
BACKUP_DIR="/home/misa/backups"

# Backup config
cp data/.config.yaml $BACKUP_DIR/config_$DATE.yaml

# Backup logs
tar -czf $BACKUP_DIR/logs_$DATE.tar.gz tmp/*.log

# Backup music library
tar -czf $BACKUP_DIR/music_$DATE.tar.gz music/

echo "Backup completed: $DATE"
```

#### 5.4.4. Security Hardening

```yaml
# Enable authentication
server:
  auth:
    enabled: true
    allowed_devices:
      - "AA:BB:CC:DD:EE:FF"  # Your ESP32 MAC address

# Generate secure auth key
import secrets
auth_key = secrets.token_hex(32)
```

### 5.5. Scaling Considerations

Nếu muốn mở rộng hệ thống:

1. **Multiple ESP32 devices** (5-10):
   - Current setup OK
   - Monitor GPU memory usage

2. **Many devices** (10-50):
   - Consider ASR API (FunASRServer)
   - Load balance with nginx

3. **Production scale** (50+):
   - Deploy Docker containers
   - Use Kubernetes for orchestration
   - Separate ASR/TTS/LLM microservices

---

## 6. QUICK REFERENCE

### 6.1. Start/Stop Commands

```bash
# Activate environment
conda activate xiaozhi

# Start server
cd ~/Desktop/RD/xiaozhi-esp32-server/main/xiaozhi-server
python app.py

# Stop server
Ctrl+C

# View logs
tail -f tmp/server.log

# Check GPU
nvidia-smi

# Monitor resources
htop
```

### 6.2. File Locations

```
~/Desktop/RD/xiaozhi-esp32-server/main/xiaozhi-server/
├── app.py                          # Entry point
├── config.yaml                     # Default config
├── data/
│   └── .config.yaml               # Your custom config ⭐
├── models/
│   ├── SenseVoiceSmall/
│   │   └── model.pt               # ASR model (400MB) ⭐
│   └── snakers4_silero-vad/       # VAD model (auto-downloaded)
├── tmp/                            # Temp audio files
│   └── server.log                 # Main log file ⭐
├── music/                          # Music library (optional)
├── core/                           # Core modules
│   ├── websocket_server.py
│   ├── connection.py
│   └── providers/
│       ├── asr/
│       ├── tts/
│       ├── llm/
│       └── ...
└── plugins_func/                   # Plugin functions
    └── functions/
```

### 6.3. Port Reference

| Port | Service | Protocol | Purpose |
|------|---------|----------|---------|
| 8000 | WebSocket | WS | Main ESP32 communication |
| 8003 | HTTP | HTTP | OTA updates, Vision API |

### 6.4. API Endpoints

```
# WebSocket
ws://YOUR_IP:8000/xiaozhi/v1/

# OTA
http://YOUR_IP:8003/xiaozhi/ota/

# Vision Analysis
http://YOUR_IP:8003/mcp/vision/explain
```

### 6.5. Useful Commands

```bash
# Find your local IP
ip addr show | grep inet

# Test WebSocket
# Use browser: main/xiaozhi-server/test/test_page.html

# Check port listening
sudo netstat -tlnp | grep :8000

# Kill process on port
sudo fof -t -i:8000
sudo kill -9 <PID>

# Disk space
df -h

# GPU memory
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

---

## 7. NEXT STEPS

### 7.1. After Successful Deployment

1. ✅ **Test basic conversation**
2. ✅ **Test function calling** (weather, news)
3. ⬜ **Customize AI personality** (edit prompt)
4. ⬜ **Add custom plugins**
5. ⬜ **Optimize for your use case**

### 7.2. Advanced Features to Explore

- **Voice cloning** với Fish-Speech
- **Home Assistant integration** cho smart home control
- **Memory system** với mem0ai
- **Vision capabilities** với camera module
- **Multi-language support**

### 7.3. Contributing

Nếu bạn phát triển thêm features hay tối ưu:

1. Fork repo
2. Tạo branch: `git checkout -b feature/your-feature`
3. Commit: `git commit -m 'Add feature'`
4. Push: `git push origin feature/your-feature`
5. Tạo Pull Request

---

## 8. SUPPORT & RESOURCES

### Official Documentation
- GitHub: https://github.com/xinnan-tech/xiaozhi-esp32-server
- Docs: https://github.com/xinnan-tech/xiaozhi-esp32-server/tree/main/docs

### Community
- Issues: https://github.com/xinnan-tech/xiaozhi-esp32-server/issues
- Discussions: https://github.com/xinnan-tech/xiaozhi-esp32-server/discussions

### External Resources
- FunASR: https://github.com/modelscope/FunASR
- Gemini API: https://ai.google.dev/gemini-api/docs
- ESP-IDF: https://docs.espressif.com/projects/esp-idf/

---

**Document Version**: 1.0  
**Last Updated**: 2025-01-25  
**Author**: AI Assistant for XiaoZhi Deployment  
**Status**: Ready for Production Testing

---

*Good luck with your deployment! 🚀*
