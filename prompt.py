from langchain.prompts import ChatPromptTemplate

def create_ocr_prompt() -> ChatPromptTemplate:
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
    "4. **Total Amount** – look for total Amount  written in indian standard  thousand  crores  lakh  hundered under labels like:\n"
    "   - 'Amount'\n"
    "   - 'Total'\n"
    "   - 'Total Charges'\n"
    "   - 'Total Amount'\n"
    "   - 'Grand Total'\n"
    "   - Any large total-related value at the end\n"
    "**Return the output in Markdown format**.\n"
    "If a field is unreadable or missing, write `(unreadable)` instead of a value.\n\n"
    "**Important:**\n"
    "- Do NOT copy values from this instruction (e.g., do not use sample dates or numbers like '5/05/25')\n"
    "- Extract only the actual data present in the image\n"
    "- Be accurate and concise"
)


    image_payload = [
        {
            "type": "image_url",
            "image_url": {
                "url": "data:image/jpeg;base64,{image_data}"
            },
        }
    ]

    return ChatPromptTemplate.from_messages([("system", system_prompt), ("user", image_payload)])
