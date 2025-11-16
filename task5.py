import tensorflow as tf # tensorflow ek deep learning framework hai 
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, decode_predictions, preprocess_input #mobileNetV2 pretrained image classification model hai  joh ImageNet par train hua hai joh k large data hai images ka ,decode_prediction model k ouput ko human readable labels me convert krta hai , preprocess_input input image ko model k liye preprocess krta hai
from tensorflow.keras.preprocessing import image #image aik helper function hai jo images ko load aur preprocess krta hai
import numpy as np #numerical calculation aur array manipulation k liay
import sys #command line arguments handle krne k liay

model = MobileNetV2(weights='imagenet')#loads model weights trained on ImagetNet dataset
print("Model loaded successfully!")

def load_and_preprocess(img_path):  
    img = image.load_img(img_path, target_size=(224, 224))  #image ko load krta hai aur uska size 224x224 pixels me resize krta hai kyun k MobileNetV2 is size k images expect krta hai
    img_array = image.img_to_array(img)   #image ko numpy array me convert krta hai
    img_array = np.expand_dims(img_array, axis=0)    #batch dimension add krta hai 0 index pr kyun k model multiple images ek sath process kr skta hai        
    img_array = preprocess_input(img_array)  #input image ko model k liye preprocess krta hai   means specific format mein convert krna jo MobileNetV2 ko dea gaya tha # we will convert image values by this formula x = x / 127.5 - 1
      
    return img_array
def classify_image(img_path):
    img_array = load_and_preprocess(img_path) #method call kiya hai image load krne k liay
    
    preds = model.predict(img_array)   #yahan model predict kryga image array ko input lekar                    
    top_preds = decode_predictions(preds, top=3)[0]        #top 3 predictions nikal raha hai starting from highest probability , batch ki pehli 1 imiage ki 

    print(f"\nTop 3 predictions for '{img_path}':")
    for i, (imagenet_id, label, prob) in enumerate(top_preds):
        print(f"{i+1}. {label} ({prob*100:.2f}%)")
if __name__ == "__main__":
    if len(sys.argv) < 2: #agar user ne image path terminal mein provide nahi kiya to
        print("Usage: python image_classifier.py <image_path>")
        sys.exit(1) #agar user kuch argument nahi deta to program exit kr jata hai with status code 1

    image_path = sys.argv[1]#image path command line argument se le rahay hai
    classify_image(image_path)#image classify krne k liay method call kr rahe hain