from typing import Optional, Any
import io
import base64
import time
import os
import json
import re
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.runnables.utils import Input, Output
from langchain_ollama import ChatOllama
from PIL import Image

from prompt import create_ocr_prompt


class OcrChain(Runnable[Input, Output]):
    def __init__(self, model: str, base_url: str, temperature: float):
        print("🔧 Initializing ChatOllama model...")
        t0 = time.time()
        self._llm = ChatOllama(
            model=model,
            base_url=base_url,
            temperature=temperature,
        )
        # ❌ DON’T pre-create prompt here anymore
        print(f"✅ Model initialized in {time.time() - t0:.2f} seconds.")

    def invoke(self, image_filename: str, config: Optional[RunnableConfig] = None, **kwargs: Any) -> dict:
        print(f"\n📂 Step 1: Reading & encoding image: {image_filename}")
        t1 = time.time()
        image_data = self._read_image(image_filename, zoom_factor=2.0)
        print(f"✅ Image processed in {time.time() - t1:.2f} seconds.")

        print("\n📦 Step 2: Creating OCR chain...")
        t2 = time.time()
        chain = self._create_chain(image_data)  # ✅ Pass image_data here
        print(f"✅ Chain created in {time.time() - t2:.2f} seconds.")

        print("\n⚡ Step 3: Invoking chain with image data...")
        t3 = time.time()
        result = chain.invoke({}, config, **kwargs)  # ✅ No need to pass image_data here
        print(f"✅ Chain invoked and response received in {time.time() - t3:.2f} seconds.")

        print("\n📄 Step 4: Final LLM output:")
        content = result.content.strip()
        print(content)

        total_time = time.time() - t1
        print(f"\n⏱️ Total processing time: {total_time:.2f} seconds.\n")

        # 🔧 Strip Markdown code block formatting if present
        if content.startswith("```json"):
            content = re.sub(r"^```json\s*|\s*```$", "", content.strip(), flags=re.DOTALL)

        # 🧠 Try parsing JSON
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            print("⚠️ LLM did not return valid JSON.")
            return {"error": "Invalid LLM output", "raw_output": content}

    def _create_chain(self, image_data: str) -> Runnable:
        print("🔗 Building the OCR prompt → LLM chain...")
        # ✅ Now dynamically create the prompt with the fresh image data
        ocr_prompt = create_ocr_prompt(image_data)
        return ocr_prompt | self._llm

    def _read_image(self, image_filename: str, zoom_factor: float = 1.5) -> str:
        print("🖼️ Loading and preprocessing image...")

        # Open and convert to RGB
        file = Image.open(image_filename).convert("RGB")

        # Resize (zoom) the image by the zoom_factor
        new_size = (int(file.width * zoom_factor), int(file.height * zoom_factor))
        file = file.resize(new_size, Image.LANCZOS)
        print(f"🔍 Image resized to: {new_size}")

        # Ensure 'output' directory exists
        output_dir = os.path.join(os.getcwd(), "output")
        os.makedirs(output_dir, exist_ok=True)

        # Save resized image to output folder
        output_path = os.path.join(output_dir, os.path.basename(image_filename))
        file.save(output_path, format="JPEG", quality=90)
        print(f"💾 Resized image saved to: {output_path}")

        # Encode the saved image to base64
        with open(output_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")

        print("✅ Image successfully encoded.")
        return encoded
