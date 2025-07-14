from typing import Optional, Any
import io
import base64
import time
import os
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
        self._ocr_prompt = create_ocr_prompt()
        print(f"✅ Model initialized and prompt created in {time.time() - t0:.2f} seconds.")

    def invoke(self, image_filename: str, config: Optional[RunnableConfig] = None, **kwargs: Any) -> str:
        print(f"\n📂 Step 1: Reading & encoding image: {image_filename}")
        t1 = time.time()
        image_data = self._read_image(image_filename)
        print(f"✅ Image processed in {time.time() - t1:.2f} seconds.")

        print("\n📦 Step 2: Creating OCR chain...")
        t2 = time.time()
        chain = self._create_chain()
        print(f"✅ Chain created in {time.time() - t2:.2f} seconds.")

        print("\n⚡ Step 3: Invoking chain with image data...")
        t3 = time.time()
        result = chain.invoke({"image_data": image_data}, config, **kwargs)
        print(f"✅ Chain invoked and response received in {time.time() - t3:.2f} seconds.")

        print("\n📄 Step 4: Final LLM output:")
        print(result.content)

        total_time = time.time() - t1
        print(f"\n⏱️ Total processing time: {total_time:.2f} seconds.\n")

        return result.content

    def _create_chain(self) -> Runnable:
        print("🔗 Building the OCR prompt → LLM chain...")
        return self._ocr_prompt | self._llm

    def _read_image(self, image_filename: str) -> str:
        print("🖼️ Loading and preprocessing image...")
        file = Image.open(image_filename).convert("RGB")
        file = file.resize((768, 768))

        # Ensure 'output' directory exists
        output_dir = os.path.join(os.getcwd(), "output")
        os.makedirs(output_dir, exist_ok=True)

        # Save resized image to output folder
        output_path = os.path.join(output_dir, os.path.basename(image_filename))
        file.save(output_path, format="JPEG", quality=90)
        print(f"💾 Image saved to: {output_path}")

        # Encode the saved image to base64
        with open(output_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")

        print("✅ Image successfully encoded.")
        return encoded
       
