from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
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
import io
import base64
import tempfile
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

load_dotenv()

if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
limiter = Limiter(key_func=get_remote_address)

app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────
# LAZY MODEL LOADING
# ─────────────────────────────────────────

_reader = None
_processor = None
_trocr_model = None
_p2t = None
_math_available = None


def get_reader():
    global _reader
    if _reader is None:
        print("Loading EasyOCR...")
        _reader = easyocr.Reader(['en'])
        print("EasyOCR loaded!")
    return _reader


def get_trocr():
    global _processor, _trocr_model
    if _processor is None:
        print("Loading TrOCR...")
        _processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-handwritten')
        _trocr_model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-handwritten')
        print("TrOCR loaded!")
    return _processor, _trocr_model


def get_p2t():
    global _p2t, _math_available
    if _math_available is None:
        try:
            from pix2text import Pix2Text
            print("Loading Pix2Text...")
            _p2t = Pix2Text.from_config()
            _math_available = True
            print("Pix2Text loaded!")
        except Exception as e:
            _math_available = False
            print(f"Pix2Text not available: {e}")
    return _p2t, _math_available


print("Server ready! Models will load on first use.")

# ─────────────────────────────────────────
# CONVERSATION MEMORY
# ─────────────────────────────────────────

conversation_histories = {}

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
    blue_ratio = blue_pixels / total_pixels

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    white_pixels = cv2.countNonZero((gray > 200).astype(np.uint8))
    white_ratio = white_pixels / total_pixels

    if white_ratio > 0.85 and blue_ratio < 0.02:
        return False
    return blue_ratio > 0.02


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
    replacements = {'|': 'I', '""': '"', '``': '"', '\x0c': ''}
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
    except Exception:
        return ""


def get_easyocr_result(processed_img):
    try:
        reader = get_reader()
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
    except Exception:
        return ""


def get_trocr_result(image_bytes):
    try:
        processor, trocr_model = get_trocr()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(img_rgb).convert("RGB")
        pixel_values = processor(pil_image, return_tensors="pt").pixel_values
        with torch.no_grad():
            generated_ids = trocr_model.generate(pixel_values)
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return text.strip()
    except Exception:
        return ""


def get_math_result(image_bytes):
    p2t, math_available = get_p2t()
    if not math_available:
        return "Pix2Text not installed. Run: pip install pix2text"
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(img_rgb)
        result = p2t.recognize(pil_image)
        return result
    except Exception as e:
        return f"Math OCR failed: {str(e)}"


def pick_best_result(trocr_text, tesseract_text, easyocr_text, handwritten):
    trocr_valid = trocr_text and len(trocr_text) > 5 and '#' not in trocr_text
    if handwritten and trocr_valid:
        return trocr_text
    else:
        if not tesseract_text and not easyocr_text:
            return "No text found in image"
        if not tesseract_text:
            return easyocr_text
        if not easyocr_text:
            return tesseract_text
        if len(tesseract_text) >= len(easyocr_text):
            return tesseract_text
        return easyocr_text


# ─────────────────────────────────────────
# TABLE EXTRACTION
# ─────────────────────────────────────────

def extract_table_from_image(image_bytes):
    """
    Uses EasyOCR with position-aware cell placement.
    No img2table dependency.
    """
    try:
        reader = get_reader()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        h, w = img.shape[:2]

        # Upscale for better OCR
        scale = max(4.0, 800 / min(h, w))
        img_large = cv2.resize(img, None, fx=scale, fy=scale,
            interpolation=cv2.INTER_CUBIC)
        ih, iw = img_large.shape[:2]
        print(f"Table image: {w}x{h} → upscaled: {iw}x{ih}")

        # Run EasyOCR on multiple color spaces
        all_detections = {}

        def add_results(img_input, label):
            try:
                results = reader.readtext(img_input)
                for result in results:
                    bbox, text, conf = result[0], result[1].strip(), result[2]
                    if not text or conf < 0.2:
                        continue
                    # Filter garbage — must be mostly alphanumeric
                    alnum = sum(1 for c in text if c.isalnum() or c in '.-')
                    if alnum < len(text) * 0.6:
                        continue
                    y_c = int((bbox[0][1] + bbox[2][1]) / 2)
                    x_c = int((bbox[0][0] + bbox[2][0]) / 2)
                    # Bucket to deduplicate nearby detections
                    key = (y_c // 20, x_c // 40)
                    if key not in all_detections or conf > all_detections[key][2]:
                        all_detections[key] = (x_c, y_c, text, conf)
                    print(f"  [{label}] '{text}' conf={conf:.2f} at ({x_c},{y_c})")
            except Exception as e:
                print(f"  [{label}] failed: {e}")

        # Pass 1: RGB
        add_results(cv2.cvtColor(img_large, cv2.COLOR_BGR2RGB), "RGB")

        # Pass 2: grayscale
        gray = cv2.cvtColor(img_large, cv2.COLOR_BGR2GRAY)
        add_results(gray, "gray")

        # Pass 3: LAB L-channel (good for colored backgrounds)
        lab_l = cv2.cvtColor(img_large, cv2.COLOR_BGR2LAB)[:, :, 0]
        add_results(lab_l, "LAB")

        # Pass 4: CLAHE enhanced
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        add_results(enhanced, "CLAHE")

        if not all_detections:
            return None, "No text detected in image"

        # Get all unique detections sorted by position
        detections = sorted(all_detections.values(), key=lambda d: (d[1], d[0]))
        print(f"\nAll detections: {[(d[2], d[0], d[1]) for d in detections]}")

        # ── Step 1: Cluster y-positions into rows ──
        y_values = [d[1] for d in detections]
        row_threshold = ih * 0.04

        row_groups = []
        current_group = [detections[0]]
        current_y = detections[0][1]

        for det in detections[1:]:
            if abs(det[1] - current_y) <= row_threshold:
                current_group.append(det)
                current_y = sum(d[1] for d in current_group) / len(current_group)
            else:
                row_groups.append(current_group)
                current_group = [det]
                current_y = det[1]
        row_groups.append(current_group)

        # Sort each row by x
        row_groups = [sorted(g, key=lambda d: d[0]) for g in row_groups]
        print(f"\nRow groups: {[[d[2] for d in g] for g in row_groups]}")

        # ── Step 2: Cluster x-positions into columns ──
        all_x = sorted(set(d[0] for g in row_groups for d in g))

        # Find column boundaries using gaps in x positions
        col_centers = []
        if all_x:
            col_centers = [all_x[0]]
            for x in all_x[1:]:
                # Check if this x is far from existing columns
                min_dist = min(abs(x - c) for c in col_centers)
                if min_dist > iw * 0.08:  # new column if far enough
                    col_centers.append(x)
                else:
                    # Merge with nearest column center
                    nearest = min(range(len(col_centers)),
                        key=lambda i: abs(col_centers[i] - x))
                    col_centers[nearest] = (col_centers[nearest] + x) / 2

        col_centers = sorted(col_centers)
        num_cols = len(col_centers)
        print(f"Column centers: {col_centers} → {num_cols} columns")

        # ── Step 3: Place each detection in correct cell ──
        def get_col_index(x):
            return min(range(num_cols),
                key=lambda i: abs(x - col_centers[i]))

        num_rows = len(row_groups)
        grid = [['' for _ in range(num_cols)] for _ in range(num_rows)]

        for row_i, group in enumerate(row_groups):
            for det in group:
                col_i = get_col_index(det[0])
                # If cell already has a value, keep higher confidence one
                if grid[row_i][col_i] == '':
                    grid[row_i][col_i] = det[2]
                else:
                    # Keep the longer/more confident value
                    if len(det[2]) > len(grid[row_i][col_i]):
                        grid[row_i][col_i] = det[2]

        print(f"\nGrid:\n")
        for row in grid:
            print(f"  {row}")

        # ── Step 4: Check if first row is header ──
        first_row = grid[0]
        is_header = any(
            not cell.replace('.', '').replace('-', '').replace(' ', '').isdigit()
            for cell in first_row if cell
        )

        if is_header and len(grid) > 1:
            headers = first_row
            data_rows = grid[1:]
        else:
            headers = [f"Col{i+1}" for i in range(num_cols)]
            data_rows = grid

        df = pd.DataFrame(data_rows, columns=headers)
        print(f"\nFinal DataFrame:\n{df}")
        return df, None

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"Table extraction failed: {str(e)}"



# ─────────────────────────────────────────
# CHART GENERATION
# ─────────────────────────────────────────

def generate_chart(df, chart_type, x_col, y_col, title=""):
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        df[y_col] = pd.to_numeric(df[y_col], errors='coerce')
        df = df.dropna(subset=[y_col])

        colors = ['#4CAF50','#2196F3','#FF9800','#E91E63',
                  '#9C27B0','#00BCD4','#FF5722','#607D8B']

        if chart_type == "bar":
            bars = ax.bar(df[x_col], df[y_col],
                color=colors[:len(df)], edgecolor='white')
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.0f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
        elif chart_type == "line":
            ax.plot(df[x_col], df[y_col], marker='o',
                color='#2196F3', linewidth=2, markersize=6)
            ax.fill_between(range(len(df[x_col])),
                df[y_col], alpha=0.1, color='#2196F3')
        elif chart_type == "pie":
            ax.pie(df[y_col], labels=df[x_col],
                autopct='%1.1f%%', colors=colors[:len(df)])
        elif chart_type == "scatter":
            ax.scatter(df[x_col], df[y_col],
                color='#9C27B0', s=100, alpha=0.7)
        elif chart_type == "histogram":
            ax.hist(df[y_col], bins=10, color='#4CAF50',
                edgecolor='white')

        chart_title = title or f"{y_col} by {x_col}"
        ax.set_title(chart_title, fontsize=14, fontweight='bold')
        if chart_type != "pie":
            ax.set_xlabel(x_col, fontsize=11)
            ax.set_ylabel(y_col, fontsize=11)
            ax.grid(True, alpha=0.3)

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        chart_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        return chart_base64
    except Exception as e:
        print(f"Chart error: {e}")
        return None


def generate_math_plot(expression, x_range=(-10, 10)):
    """Generate a plot for a mathematical function"""
    try:
        import numpy as np
        x = np.linspace(x_range[0], x_range[1], 500)

        # Safe eval for math expressions
        safe_dict = {
            'x': x, 'np': np,
            'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
            'exp': np.exp, 'log': np.log, 'sqrt': np.sqrt,
            'pi': np.pi, 'e': np.e, 'abs': np.abs
        }

        y = eval(expression, {"__builtins__": {}}, safe_dict)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x, y, color='#2196F3', linewidth=2)
        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.axvline(x=0, color='black', linewidth=0.5)
        ax.grid(True, alpha=0.3)
        ax.set_title(f"f(x) = {expression}", fontsize=14, fontweight='bold')
        ax.set_xlabel('x', fontsize=11)
        ax.set_ylabel('f(x)', fontsize=11)
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        chart_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        return chart_base64
    except Exception as e:
        return None


# ─────────────────────────────────────────
# AI PROCESSING
# ─────────────────────────────────────────

def process_with_ai(text, mode, language="Hindi"):
    prompts = {
        "summarize": "Summarize the following text clearly and concisely:",
        "fix": "Fix any spelling or grammar errors and return corrected version only:",
        "explain": "Explain the following text in simple words as if explaining to a student:",
        "solve": "Solve the following questions or problems and show the steps clearly:",
        "bullet": "Convert the following text into clear bullet points:",
        "translate": f"Translate the following text to {language}:",
        "math": """You are a university level math professor.
The following is a mathematical expression or equation extracted from an image.
Please:
1. First rewrite the equation clearly in LaTeX
2. Solve it step by step showing all working
3. State the final answer clearly in LaTeX
4. Briefly explain each step

Expression:""",
    }
    instruction = prompts.get(mode, "Process the following text:")
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": f"{instruction}\n\n{text}"}]
        )
        return response.choices[0].message.content
    except Exception as groq_error:
        return (f"⚠️ AI service temporarily unavailable.\n\n"
                f"**Extracted text:**\n{text}\n\nPlease try again in a moment.")


def smart_chat(message, history, ocr_text, table_csv=""):
    """
    Smart chat that decides response type based on user message.
    Returns: {type, content}
    type can be: text, table, chart, math, mixed
    """

    system_message = """You are a smart AI assistant that can answer questions about 
extracted text and data. You MUST respond in JSON format only.

Based on the user's question, decide the best response type:
- "text": for explanations, summaries, general answers
- "table": when user asks for tabular data, statistics, comparisons
- "chart": when user asks for graphs, charts, visualizations
- "math": when answer involves mathematical equations
- "mixed": for combinations

Response format (JSON only, no other text):
{
  "type": "text|table|chart|math|mixed",
  "text": "markdown text answer (always include this)",
  "table": {
    "headers": ["col1", "col2"],
    "rows": [["val1", "val2"]]
  },
  "chart": {
    "type": "bar|line|pie|scatter",
    "x_col": "column name",
    "y_col": "column name",
    "title": "chart title"
  },
  "math_expression": "numpy compatible expression for plotting e.g. np.sin(x)"
}

Only include fields that are needed.
For table type: always include table field with headers and rows.
For chart type: include chart field with specifications.
For math type: include LaTeX in text field and optionally math_expression for plotting.
"""

    context = ""
    if ocr_text:
        context += f"\n\nExtracted text from image:\n{ocr_text}"
    if table_csv:
        context += f"\n\nTable data (CSV):\n{table_csv}"

    messages = [{"role": "system", "content": system_message}]

    # Add history
    for h in history:
        if h.get("role") in ["user", "assistant"]:
            messages.append({"role": h["role"], "content": h["content"]})

    messages.append({
        "role": "user",
        "content": f"{message}{context}"
    })

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            max_tokens=2048
        )
        raw = response.choices[0].message.content.strip()

        # Clean JSON
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        result = json.loads(raw)
        return result

    except json.JSONDecodeError:
        return {"type": "text", "text": raw if 'raw' in dir() else "Sorry, could not process your request."}
    except Exception as e:
        return {"type": "text", "text": f"Chat failed: {str(e)}"}


# ─────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────

@app.post("/ocr")
@limiter.limit("10/minute")
async def ocr(request: Request, file: UploadFile = File(...)):
    allowed_types = ["image/jpeg", "image/png", "image/jpg", "image/webp"]
    if file.content_type not in allowed_types:
        return {"error": "Invalid file type. Please upload JPG, PNG or WebP."}

    image_bytes = await file.read()
    if len(image_bytes) > 10 * 1024 * 1024:
        return {"error": "File too large. Maximum size is 10MB."}

    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return {"error": "Could not read image. Please upload a valid image file."}

    handwritten = is_handwritten(img)
    processed = preprocess_image(image_bytes)

    tesseract_text = get_tesseract_result(processed)
    easyocr_text = get_easyocr_result(processed)
    trocr_text = get_trocr_result(image_bytes)
    math_text = get_math_result(image_bytes)

    # Try table extraction on same image
    df, table_error = extract_table_from_image(image_bytes)
    table_data = None
    if df is not None:
        print(f"Table detected! Columns: {df.columns.tolist()}")
        print(f"Table data:\n{df}")
        table_data = {
            "columns": df.columns.tolist(),
            "data": df.values.tolist(),
            "csv": df.to_csv(index=False)
        }
    else:
        print(f"No table detected. Error: {table_error}")

    best_text = pick_best_result(
        trocr_text, tesseract_text, easyocr_text, handwritten
    )
    cleaned = clean_text(best_text)

    return {
        "text": cleaned,
        "tesseract": clean_text(tesseract_text),
        "easyocr": clean_text(easyocr_text),
        "trocr": trocr_text,
        "math": math_text,
        "handwritten": handwritten,
        "table": table_data
    }


@app.post("/ai-process")
@limiter.limit("20/minute")
async def ai_process(
    request: Request,
    text: str = Form(...),
    mode: str = Form(...),
    language: str = Form(default="Hindi")
):
    if not text.strip():
        return {"result": "No text to process"}
    result = process_with_ai(text, mode, language)
    return {"result": result}


@app.post("/chat")
@limiter.limit("30/minute")
async def chat(
    request: Request,
    message: str = Form(...),
    session_id: str = Form(default="default"),
    ocr_text: str = Form(default=""),
    table_csv: str = Form(default="")
):
    if not message.strip():
        return {"type": "text", "text": "Please enter a message.", "history_length": 0}

    if session_id not in conversation_histories:
        conversation_histories[session_id] = []

    history = conversation_histories[session_id]

    # Get smart response
    result = smart_chat(message, history, ocr_text, table_csv)

    # If chart requested and we have table data, generate it
    if result.get("type") in ["chart", "mixed"] and result.get("chart") and table_csv:
        try:
            df = pd.read_csv(io.StringIO(table_csv))
            chart_spec = result["chart"]
            chart_b64 = generate_chart(
                df,
                chart_spec.get("type", "bar"),
                chart_spec.get("x_col", df.columns[0]),
                chart_spec.get("y_col", df.columns[1] if len(df.columns) > 1 else df.columns[0]),
                chart_spec.get("title", "")
            )
            result["chart_image"] = chart_b64
        except Exception as e:
            result["chart_error"] = str(e)

    # If math plot requested
    if result.get("math_expression"):
        chart_b64 = generate_math_plot(result["math_expression"])
        if chart_b64:
            result["chart_image"] = chart_b64

    # Save to history
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": result.get("text", "")})

    if len(history) > 20:
        conversation_histories[session_id] = history[-20:]

    result["history_length"] = len(history)
    return result


@app.post("/generate-chart")
@limiter.limit("10/minute")
async def generate_chart_endpoint(
    request: Request,
    csv_data: str = Form(...),
    chart_type: str = Form(...),
    x_col: str = Form(...),
    y_col: str = Form(...),
    title: str = Form(default="")
):
    try:
        df = pd.read_csv(io.StringIO(csv_data))
        chart = generate_chart(df, chart_type, x_col, y_col, title)
        if chart:
            return {"chart": chart}
        return {"error": "Could not generate chart"}
    except Exception as e:
        return {"error": str(e)}


@app.post("/export-excel")
@limiter.limit("10/minute")
async def export_excel(
    request: Request,
    csv_data: str = Form(...)
):
    try:
        df = pd.read_csv(io.StringIO(csv_data))
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Data')
        output.seek(0)
        return StreamingResponse(
            output,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": "attachment; filename=extracted_data.xlsx"}
        )
    except Exception as e:
        return {"error": str(e)}


@app.post("/clear-chat")
async def clear_chat(session_id: str = Form(default="default")):
    if session_id in conversation_histories:
        conversation_histories[session_id] = []
    return {"status": "cleared"}