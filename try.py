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
         "You are a precise OCR assistant. Carefully extract the following fields from the provided invoice image. "
    "The invoice may include both printed and handwritten text.\n\n"
    "**Extract the following fields only:**\n"
    "1. **Vendor Name** – the name of the shop or company at the top of the invoice\n"
    "2. **Invoice Number** – typically labeled as 'No.', 'Invoice No.', or similar\n"
    "3. **Invoice Date** – this may be handwritten or printed. Look for patterns like:\n"
    "   - DD-MM-YYYY \n"
    "   - DD/MM/YYYY \n"
    "   - D-M-YY \n"
    "   - Also check near the word 'Date' or 'Dated'\n"
    "4. **Total Amount** – look for total charges under labels like:\n"
    "   - 'Amount'\n"
    "   - 'Total'\n"
    "   - 'Total Charges'\n"
    "   - 'Vehicle Charges'\n"
    "   - 'Hotel Charges'\n"
    "   - 'Balance to be Paid'\n"
    "   - 'Total Amount'\n"
    "   - Any large total-related value at the end\n"
    "**Return the output in Markdown format**.\n"
    "If a field is unreadable or missing, write `(unreadable)` instead of a value.\n\n"
    "**Important:**\n"
    "- Do NOT copy values from this instruction (e.g., do not use sample dates or numbers like '5/05/25')\n"
    "- Extract only the actual data present in the image\n"
    "- Be accurate and concise"

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
    img_path = 'test.jpeg'
    text = extract_text(img_path)
    print(text)
    output = run_llm(text)
    print(output.content)
