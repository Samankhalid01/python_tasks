import sys
from PIL import Image
import torch
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

def load_model(device):
    model_name = "nlpconnect/vit-gpt2-image-captioning"
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    feature_extractor = ViTImageProcessor.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.to(device)
    return model, feature_extractor, tokenizer

def generate_caption(image_path, model, feature_extractor, tokenizer, device):
    image = Image.open(image_path).convert("RGB")
    pixel_values = feature_extractor(images=[image], return_tensors="pt").pixel_values.to(device)

    # generation params (simple)
    gen_kwargs = {"max_length": 32, "num_beams": 4, "no_repeat_ngram_size": 2, "early_stopping": True}
    output_ids = model.generate(pixel_values, **gen_kwargs)
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    return caption

def main():
    if len(sys.argv) < 2:
        print("Usage: python task9_image_caption.py path/to/image.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model, feature_extractor, tokenizer = load_model(device)
    caption = generate_caption(image_path, model, feature_extractor, tokenizer, device)
    print("Predicted caption:")
    print(caption)

if __name__ == "__main__":
    main()
