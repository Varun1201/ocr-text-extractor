import pytesseract
import cv2
import numpy as np
from PIL import Image

# Point to tesseract installation
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def preprocess_image(image_path):
    # Read image
    img = cv2.imread(image_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Remove noise
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    
    # Increase contrast
    contrast = cv2.equalizeHist(denoised)
    
    # Thresholding - make text black, background white
    _, thresh = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return thresh

def extract_text(image_path):
    # Preprocess
    processed = preprocess_image(image_path)
    
    # Convert to PIL Image for tesseract
    pil_image = Image.fromarray(processed)
    
    # Extract text
    text = pytesseract.image_to_string(pil_image, config='--psm 7 --oem 3')
    
    return text.strip()

if __name__ == "__main__":
    # Test with an image
    image_path = "test.jpg"  # put your image path here
    text = extract_text(image_path)
    print("Extracted Text:")
    print("-" * 40)
    print(text)