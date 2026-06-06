# OCR + AI Text Extractor

Convert photos of printed and handwritten text into digital text
using multiple OCR engines, then process with AI.

## Features
- Multi-engine OCR (Tesseract, EasyOCR, TrOCR)
- Automatic handwritten vs printed text detection
- Smart engine selection — chooses best result automatically
- Select which OCR engine's text to send to AI
- AI powered post processing using Groq (Llama 3):
  - Summarize text
  - Fix grammar and spelling
  - Explain in simple words
  - Solve questions and problems
  - Convert to bullet points
  - Translate to Hindi

## Tech Stack
- FastAPI (backend)
- TrOCR — Microsoft transformer model (handwriting)
- EasyOCR (handwriting + printed)
- Tesseract (printed text)
- OpenCV (image preprocessing)
- Groq AI — Llama 3 (AI post processing)
- HTML/CSS/JavaScript (frontend)
- Docker (containerization)

## How to run locally

### 1. Install Tesseract (Windows)
Download from: https://github.com/UB-Mannheim/tesseract/wiki

### 2. Clone the repo
```bash
git clone https://github.com/Varun1201/ocr-text-extractor.git
cd ocr-text-extractor
```

### 3. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

### 5. Set up environment variables
Create a `.env` file in the project root:
GROQ_API_KEY=your-groq-api-key-here
Get a free Groq API key at: https://console.groq.com

### 6. Start the API
```bash
uvicorn app:app --reload
```

### 7. Open frontend
Open `index.html` in your browser

## How to run with Docker
```bash
docker build -t ocr-text-extractor .
docker run -p 8000:8000 ocr-text-extractor
```

## How it works
1. Upload a photo of printed or handwritten text
2. App automatically detects if text is handwritten or printed
3. OpenCV preprocesses the image accordingly
4. Three OCR engines analyze it:
   - TrOCR for handwriting
   - EasyOCR for mixed content
   - Tesseract for printed text
5. Best result selected automatically based on content type
6. All results shown in tabbed interface
7. Select any OCR engine tab and process with AI:
   - Summarize, Fix Grammar, Explain, Solve, Bullet Points, Translate

## Project Structure
├── app.py              # FastAPI backend with all OCR engines and AI
├── ocr.py              # OCR utility functions
├── index.html          # Frontend UI with tabbed results and AI section
├── requirements.txt    # Python dependencies
├── Dockerfile          # Docker configuration
└── tests/
    └── test_ocr.py     # Pytest unit tests

## API Endpoints
| Endpoint | Method | Description |
|---|---|---|
| `/ocr` | POST | Extract text from image |
| `/ai-process` | POST | Process extracted text with AI |

## Environment Variables
| Variable | Description |
|---|---|
| `GROQ_API_KEY` | Groq API key for AI processing |
