from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from ocr import OcrChain
import base64
import tempfile
import os
import uuid
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ocr_chain = OcrChain(
    model="gemma3:4b-it-q4_K_M",
    base_url="http://localhost:11434",
    temperature=0.0,
)

@app.websocket("/ws/ocr")
async def websocket_ocr(websocket: WebSocket):
    await websocket.accept()
    try:
        data = await websocket.receive_text()
        body = json.loads(data)
        image_b64 = body.get("image")

        if not image_b64:
            await websocket.send_text(json.dumps({"status": "error", "message": "No image data received."}))
            return

        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as f:
            f.write(base64.b64decode(image_b64))
            temp_path = f.name

        # Run OCR
        result = ocr_chain.invoke(temp_path)

        # Clean up
        os.remove(temp_path)

        await websocket.send_text(json.dumps({
            "status": "success",
            "result": {
                "invoiceNo": result.get("Invoice Number", ""),
                "vendorName": result.get("Vendor Name", ""),
                "invoiceDate": result.get("Invoice Date", ""),
                "amount": result.get("Invoice Amount", ""),
            }
        }))

    except Exception as e:
        await websocket.send_text(json.dumps({"status": "error", "message": str(e)}))

    finally:
        await websocket.close()
