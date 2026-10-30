# Production Deployment Guide

## 1. Local Environment Execution
```bash
# Clone the repository
git clone git@github.com:Chanakya-GH/fin-ai.git
cd fin-ai

# Activate Python 3.11 virtual environment
source venv/bin/activate

# Start FastAPI production server
DEMO_MODE=true uvicorn src.app:app --host 0.0.0.0 --port 8000
```
Open `http://localhost:8000` for the Web Dashboard or `http://localhost:8000/docs` for OpenAPI docs.

---

## 2. Docker Container Deployment

### Build the Docker Image
```bash
docker build -t fin-advisory-agent:latest .
```

### Run Container
```bash
docker run -d -p 8000:8000 --name fin-ai fin-advisory-agent:latest
```

---

## 3. CPU GGUF Quantization & Ollama Deployment (Roadmap Section 5.1)

### Merge Adapter & Export
```bash
# Generate Modelfile
python src/export_gguf.py --generate-modelfile
```

### Create Ollama Model
```bash
ollama create fin-advisory-agent -f Modelfile
ollama run fin-advisory-agent "What was Gilead's net income in 2023?"
```
