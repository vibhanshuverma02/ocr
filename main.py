from argparse import ArgumentParser
from ocr import OcrChain
import time

def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="gemma3:4b-it-q4_K_M",
        help="The model to use for the chat.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:11434",
        help="The base URL of the Ollama server.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="The temperature to use for the chat.",
    )
    parser.add_argument(
        "--input-image",
        type=str,
        required=True,
        help="The image to perform OCR on.",
    )
    args = parser.parse_args()

    print("\n🚀 Starting OCR + LLM pipeline...")
    overall_start = time.time()

    print("🔧 Step 1: Initialize OcrChain...")
    t1 = time.time()
    ocr_chain = OcrChain(
        model=args.model,
        base_url=args.base_url,
        temperature=args.temperature,
    )
    print(f"✅ OcrChain initialized in {time.time() - t1:.2f} seconds.\n")

    print("📂 Step 2: Running OCR + LLM inference...")
    t2 = time.time()
    result = ocr_chain.invoke(args.input_image)
    print(f"✅ OCR + LLM inference done in {time.time() - t2:.2f} seconds.\n")

    print("📄 Final OCR result from LLM:")
    print(result)

    total_time = time.time() - overall_start
    print(f"\n⏱️ Total pipeline time: {total_time:.2f} seconds.\n")

if __name__ == "__main__":
    main()
