import cv2
import easyocr
from PIL import Image
import requests
import json

def crop_region(image_path, coords, save_path):
    """
    Crop a region from the image.
    coords = (x, y, w, h)
    """
    img = cv2.imread(image_path)
    x, y, w, h = coords
    crop = img[y:y+h, x:x+w]
    cv2.imwrite(save_path, crop)
    return save_path

def ocr_image(image_path, gpu=False):
    """
    Run OCR (EasyOCR) on an image and return extracted text.
    """
    reader = easyocr.Reader(['en'], gpu=gpu)
    result = reader.readtext(image_path, detail=0)
    return "\n".join(result)

def call_llm(ocr_text):
    """
    Send prompt with OCR text to LLM (e.g., Ollama) and get structured JSON.
    """
    prompt = f"""
You are an invoice parser. Extract:
- vendor_name
- invoice_no
- invoice_date
- invoice_amount
from the text below.

OCR text:
{ocr_text}

Return JSON only.
"""
    # Example using Ollama API
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3",   # replace with your local model name
            "prompt": prompt
        }
    )
    result = response.json()
    return result['response']

if __name__ == "__main__":
    image_path = ""   # replace with your actual image path

    print("✅ Step 1: OCR printed parts...")
    full_text = ocr_image(image_path, gpu=False)
    print("\n📝 OCR printed text:\n", full_text)

    print("\n✂ Step 2: Crop handwritten regions and OCR...")
    # ⚠ Coordinates are examples; adjust for your template:
    fields = {
        "invoice_no": (50, 120, 200, 50),     # x, y, w, h
        "invoice_date": (900, 120, 200, 50),
        "invoice_amount": (400, 600, 250, 50)
    }
    cropped_texts = {}
    for field, coords in fields.items():
        cropped_img = f"crop_{field}.jpg"
        crop_region(image_path, coords, cropped_img)
        text = ocr_image(cropped_img, gpu=False)
        cropped_texts[field] = text.strip()

    print("\n📝 OCR cropped handwritten fields:\n", json.dumps(cropped_texts, indent=2))

    print("\n🤖 Step 3: Combine and call LLM...")
    combined_text = f"{full_text}\n" + "\n".join([f"{k}: {v}" for k,v in cropped_texts.items()])
    structured = call_llm(combined_text)

    print("\n✅ LLM extracted fields:\n", structured)
