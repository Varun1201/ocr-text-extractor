from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import pytesseract
import easyocr
import cv2
import numpy as np
import torch
import platform
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Groq setup
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading EasyOCR...")
reader = easyocr.Reader(['en'])

print("Loading TrOCR...")
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-handwritten')
trocr_model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-handwritten')

print("All models loaded!")


# ─────────────────────────────────────────
# IMAGE PROCESSING
# ─────────────────────────────────────────

def is_handwritten(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    blue_pixels = cv2.countNonZero(blue_mask)
    total_pixels = img.shape[0] * img.shape[1]
    return (blue_pixels / total_pixels) > 0.01


def preprocess_image(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if is_handwritten(img):
        img = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([90, 50, 50])
        upper_blue = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, dark_mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
        combined_mask = cv2.bitwise_or(blue_mask, dark_mask)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        detected_lines = cv2.morphologyEx(
            combined_mask, cv2.MORPH_OPEN, horizontal_kernel, iterations=2
        )
        combined_mask = cv2.subtract(combined_mask, detected_lines)
        kernel = np.ones((2, 2), np.uint8)
        dilated = cv2.dilate(combined_mask, kernel, iterations=1)
        return cv2.bitwise_not(dilated)
    else:
        img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray, h=10)
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened = cv2.filter2D(denoised, -1, kernel)
        thresh = cv2.adaptiveThreshold(
            sharpened, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 31, 10
        )
        return thresh


# ─────────────────────────────────────────
# OCR ENGINES
# ─────────────────────────────────────────

def clean_text(text):
    if not text:
        return ""
    replacements = {
        '|': 'I',
        '""': '"',
        '``': '"',
        '\x0c': '',
    }
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if line:
            for wrong, correct in replacements.items():
                line = line.replace(wrong, correct)
            cleaned_lines.append(line)
    return '\n'.join(cleaned_lines)


def get_tesseract_result(processed_img):
    try:
        pil_image = Image.fromarray(processed_img)
        text = pytesseract.image_to_string(
            pil_image, config='--psm 6 --oem 3'
        ).strip()
        return text
    except:
        return ""


def get_easyocr_result(processed_img):
    try:
        results = reader.readtext(processed_img)
        if not results:
            return ""
        results = sorted(results, key=lambda x: x[0][0][1])
        lines = []
        current_line = []
        current_y = results[0][0][0][1]
        line_threshold = 20
        for result in results:
            y_pos = result[0][0][1]
            if abs(y_pos - current_y) > line_threshold:
                if current_line:
                    current_line = sorted(current_line, key=lambda x: x[0][0][0])
                    lines.append(" ".join([r[1] for r in current_line]))
                current_line = [result]
                current_y = y_pos
            else:
                current_line.append(result)
        if current_line:
            current_line = sorted(current_line, key=lambda x: x[0][0][0])
            lines.append(" ".join([r[1] for r in current_line]))
        return '\n'.join(lines).strip()
    except:
        return ""


def get_trocr_result(image_bytes):
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(img_rgb).convert("RGB")
        pixel_values = processor(pil_image, return_tensors="pt").pixel_values
        with torch.no_grad():
            generated_ids = trocr_model.generate(pixel_values)
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return text.strip()
    except:
        return ""


def pick_best_result(trocr_text, tesseract_text, easyocr_text, handwritten):
    if handwritten:
        if trocr_text:
            return trocr_text
        return easyocr_text or tesseract_text or "No text found"
    else:
        if not tesseract_text and not easyocr_text:
            return "No text found in image"
        if not tesseract_text:
            return easyocr_text
        if not easyocr_text:
            return tesseract_text
        if len(easyocr_text) >= len(tesseract_text):
            return easyocr_text
        return tesseract_text


# ─────────────────────────────────────────
# AI PROCESSING
# ─────────────────────────────────────────

def process_with_ai(text, mode):
    prompts = {
        "summarize": "Summarize the following text clearly and concisely:",
        "fix": "Fix any spelling or grammar errors and return corrected version only:",
        "explain": "Explain the following text in simple words as if explaining to a student:",
        "solve": "Solve the following questions or problems and show the steps clearly:",
        "bullet": "Convert the following text into clear bullet points:",
        "translate": "Translate the following text to Hindi:",
    }
    instruction = prompts.get(mode, "Process the following text:")
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "user",
                    "content": f"{instruction}\n\n{text}"
                }
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"AI processing failed: {str(e)}"


# ─────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────

@app.post("/ocr")
async def ocr(file: UploadFile = File(...)):
    image_bytes = await file.read()

    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    handwritten = is_handwritten(img)

    processed = preprocess_image(image_bytes)

    tesseract_text = get_tesseract_result(processed)
    easyocr_text = get_easyocr_result(processed)
    trocr_text = get_trocr_result(image_bytes)

    best_text = pick_best_result(
        trocr_text, tesseract_text, easyocr_text, handwritten
    )
    cleaned = clean_text(best_text)

    return {
        "text": cleaned,
        "tesseract": clean_text(tesseract_text),
        "easyocr": clean_text(easyocr_text),
        "trocr": trocr_text,
        "handwritten": handwritten
    }


@app.post("/ai-process")
async def ai_process(text: str = Form(...), mode: str = Form(...)):
    if not text.strip():
        return {"result": "No text to process"}
    result = process_with_ai(text, mode)
    return {"result": result}