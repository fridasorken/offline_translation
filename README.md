# Offline Translation System

Offline translation evaluation system with a FastAPI backend and Streamlit frontend.

## Architecture

- **Backend** (`src/eval_api`): FastAPI server with pluggable translation model adapters
- **Frontend** (`src/frontend`): Streamlit web interface for translation requests

## Quick Start

### 1. Start the Backend

```bash
cd src/eval_api
uv sync
uv run uvicorn app.main:app --reload
```

The backend API will be available at `http://localhost:8000`.
API documentation is available at `http://localhost:8000/docs`.

### 2. Start the Frontend

In a new terminal:

```bash
cd src/frontend
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

# configure to use the backend
export USE_MOCK=false
export API_BASE_URL=http://localhost:8000

streamlit run app.py
```

The frontend will be available at `http://localhost:8501`.

## Features

- **Dynamic model loading**: Frontend automatically fetches available models from backend
- **Real-time translation**: Send text for translation and receive results with latency metrics
- **Language pair validation**: Backend validates supported language pairs per model
- **Mock mode**: Test frontend without backend using `USE_MOCK=true`

## Configuration

### Backend Configuration

Models are configured in `src/eval_api/models.json`. Example:

```json
{
  "opus-mt-en-gmq": {
    "adapter": "transformers",
    "model_path": "Helsinki-NLP/opus-mt-en-gmq",
    "supported_pairs": [["en", "nob"], ["en", "nno"]],
    "device": "cpu",
    "local_files_only": false
  }
}
```

### Frontend Configuration

Environment variables:

- `API_BASE_URL`: Backend API URL (default: `http://localhost:8000`)
- `USE_MOCK`: Use mock translation (default: `true`)

## API Endpoints

### POST /translate

Translate text using a specified model.

**Request:**
```json
{
  "src_lang": "en",
  "tgt_lang": "nob",
  "source": "Hello world",
  "model_id": "opus-mt-en-gmq"
}
```

**Response:**
```json
{
  "model_id": "opus-mt-en-gmq",
  "translated_value": "Hei verden",
  "latency_ms": 147
}
```

### GET /models

List all available models and their supported language pairs.

**Response:**
```json
{
  "models": [
    {
      "model_id": "opus-mt-en-gmq",
      "adapter": "transformers",
      "supported_pairs": [["en", "nob"], ["en", "nno"]]
    }
  ]
}
```

## Development

See individual README files for more details:
- Backend: `src/eval_api/README.md`
- Frontend: `src/frontend/README.md`
