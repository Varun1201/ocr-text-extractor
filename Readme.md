# OCR Text Extractor

Converts photos of printed and handwritten text into digital text 
using multiple OCR engines.

## Tech Stack
- FastAPI (backend)
- TrOCR — Microsoft transformer model (handwriting)
- EasyOCR (handwriting + printed)
- Tesseract (printed text)
- OpenCV (image preprocessing)
- HTML/CSS/JavaScript (frontend)
- Docker (containerization)

## How to run locally

### 1. Install Tesseract (Windows)
Download from: https://github.com/UB-Mannheim/tesseract/wiki

### 2. Clone the repo
git clone https://github.com/Varun1201/ocr-text-extractor.git
cd ocr-text-extractor

### 3. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

### 4. Install dependencies
pip install -r requirements.txt

### 5. Start the API
uvicorn app:app --reload

### 6. Open frontend
Open index.html in your browser

## How to run with Docker
docker build -t ocr-text-extractor .
docker run -p 8000:8000 ocr-text-extractor

## How it works
1. Upload a photo of text
2. OpenCV preprocesses the image
3. Three OCR engines analyze it:
   - TrOCR for handwriting
   - EasyOCR for mixed content  
   - Tesseract for printed text
4. Best result is selected automatically
5. All results shown in tabbed interface

## Project Structure
├── app.py          # FastAPI backend with all OCR engines
├── ocr.py          # OCR utility functions
├── index.html      # Frontend UI
├── requirements.txt
├── Dockerfile
└── tests/
    └── test_ocr.py