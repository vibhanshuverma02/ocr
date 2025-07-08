from argparse import ArgumentParser
from ocr import OcrChain

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
    
    ocr_chain = OcrChain(
        model=args.model,
        base_url=args.base_url,
        temperature=args.temperature,
    )
    result = ocr_chain.invoke(args.input_image)

    print("OCR result:", result)

if __name__ == "__main__":
    main()
