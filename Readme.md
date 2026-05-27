# OCR Text Extractor

Converts photos of printed text into digital text using Tesseract and OpenCV.

## Tech Stack
- FastAPI (backend)
- Tesseract OCR (text recognition)
- OpenCV (image preprocessing)
- HTML/CSS/JavaScript (frontend)

## How to run

### 1. Install Tesseract
Download from: https://github.com/UB-Mannheim/tesseract/wiki

### 2. Install dependencies
pip install -r requirements.txt

### 3. Start the API
uvicorn app:app --reload

### 4. Open frontend
Open index.html in your browser

## How it works
1. Upload a photo of printed text
2. OpenCV preprocesses the image (grayscale, denoise, threshold)
3. Tesseract reads the text
4. Result displayed on screen with copy option