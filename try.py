# ocr_pipeline.py
from pytesseract import image_to_string
from PIL import Image
import base64
import io
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate

def extract_text(image_path: str) -> str:
    image = Image.open(image_path)
    return image_to_string(image)

def create_prompt(raw_text: str) -> ChatPromptTemplate:
    system_prompt = (
        "Extract these fields from the invoice text:\n"
        "1. Vendor Name\n"
        "2. Invoice Number\n"
        "3. Invoice Date\n"
        "4. Item Table (Description, Quantity, Rate, Amount)\n"
        "5. Total Amount\n\n"
        "Return only the values in Markdown. If a field is missing, write (unreadable)."
    )
    return ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", raw_text)
    ])

def run_llm(raw_text: str):
    llm = ChatOllama(
        model="llava:7b-v1.5-q4_K_M",
        base_url="http://localhost:11434",
        temperature=0
    )
    chain = create_prompt(raw_text) | llm
    return chain.invoke({})

if __name__ == "__main__":
    img_path = r'D:\downloads\test.jpeg'
    text = extract_text(img_path)
    output = run_llm(text)
    print(output.content)
