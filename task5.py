import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, decode_predictions, preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
import sys

model = MobileNetV2(weights='imagenet')
print("Model loaded successfully!")

model = MobileNetV2(weights='imagenet')
print("Model loaded successfully!")
def load_and_preprocess(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  
    img_array = image.img_to_array(img)                     
    img_array = np.expand_dims(img_array, axis=0)            
    img_array = preprocess_input(img_array)                 
    return img_array
def classify_image(img_path):
    img_array = load_and_preprocess(img_path)
    
    preds = model.predict(img_array)                       
    top_preds = decode_predictions(preds, top=3)[0]        

    print(f"\nTop 3 predictions for '{img_path}':")
    for i, (imagenet_id, label, prob) in enumerate(top_preds):
        print(f"{i+1}. {label} ({prob*100:.2f}%)")
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python image_classifier.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    classify_image(image_path)
