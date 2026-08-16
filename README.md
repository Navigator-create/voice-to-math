# 🎤 Voice Math Calculator for Marine Engineers

## The Problem

Marine engineers work in noisy engine rooms with greasy, oil-covered hands. Touching keyboards, touchscreens, or keypads is impractical — and unsafe when handling heavy machinery. Yet they constantly need rapid calculations:

- Fuel consumption at specific RPM
- Ballast trim adjustments
- Electrical load balancing
- RPM-to-speed conversions
- Tank volume calculations
- Generator load calculations

**This app lets engineers speak calculations hands-free and get instant answers.**

---

## How It Works

1. Engineer presses the microphone button and speaks naturally: *"two hundred times fifteen point three"*
2. Browser captures audio via Web Audio API (WAV, 16kHz mono)
3. FastAPI backend sends audio to OpenAI Whisper API for transcription
4. Custom AST-based evaluator parses and computes the result safely
5. Result displays instantly — no typing, no touching

---

## Technical Stack

| Component | Technology |
|-----------|------------|
| Backend | Python 3.11+, FastAPI, Uvicorn |
| Frontend | Vanilla JavaScript (no frameworks) |
| Speech-to-Text | OpenAI Whisper API (cloud) |
| Math Parser | Custom `ast.literal_eval` with restricted globals |
| Environment | python-dotenv for secrets management |
| Audio Capture | Web Audio API (browser-native) |

---

## Current Status

✅ Fully functional on `localhost:8000`  
✅ Environment variables secured via `.gitignore`  
✅ Production-ready for local/dockside testing  
✅ Pluggable architecture for offline migration

---

## Deployment Constraints & Mitigations

**Current Implementation:** Uses OpenAI Whisper API (cloud-based speech-to-text). This is acceptable for:
- Dockside testing / shipyard trials
- Vessels with reliable satellite/4G connectivity
- Shore-based engineering offices

**Future Production Deployment:** For deep-sea vessels with limited/unreliable internet, I will replace Whisper API with:
- **Faster-Whisper** (CTranslate2) running locally on shipboard PC
- **ONNX runtime** for CPU inference on industrial hardware
- **Fallback keyword matcher** (10-20 common phrases) when network drops

**Architectural Decision:** The FastAPI endpoint accepts audio files and returns transcriptions via a **pluggable interface**. Swapping cloud for local model requires only changing the `transcribe()` implementation — no UI or business logic changes.

---

## Security

- All API keys stored in `.env` (excluded from version control)
- Math evaluator uses `ast.literal_eval` with strict type checking — **no `eval()` on raw strings**
- CORS disabled for production; localhost only
- Input validation sanitizes all arithmetic expressions

---

## Local Setup

```bash
# Clone the repository
git clone https://github.com/Navigator-create/voice-to-math.git
cd voice-to-math

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# Run the application
uvicorn main:app --reload --host 0.0.0.0 --port 8000
