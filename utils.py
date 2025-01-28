
import PIL.Image
import cv2
import google.generativeai as genai
from ultralytics import YOLO

# Initialize the generative AI model
GOOGLE_API_KEY = ''
genai.configure(api_key=GOOGLE_API_KEY)
model_gemini = genai.GenerativeModel(model_name="gemini-2.0-flash-exp")

model_yolo = YOLO('Yolo11n_Finetuned.pt')

labels = ['shirt, blouse', 'top, t-shirt, sweatshirt', 'sweater', 'cardigan', 'jacket', 'vest', 'pants', 'shorts', 'skirt', 'coat', 'dress', 'jumpsuit', 'cape', 
          'glasses', 'hat', 'headband, head covering, hair accessory', 'tie', 'glove', 'watch', 'belt', 'leg warmer', 'tights, stockings', 'sock', 'shoe', 'bag, wallet', 
          'scarf', 'umbrella', 'hood', 'collar', 'lapel', 'epaulette', 'sleeve', 'pocket', 'neckline', 'buckle', 'zipper', 'applique', 'bead', 'bow', 'flower', 'fringe', 
          'ribbon', 'rivet', 'ruffle', 'sequin', 'tassel']

upper = ["shirt, blouse","top, t-shirt, sweatshirt","sweater","cardigan","jacket","vest","coat","dress","jumpsuit","cape","hood"]

lower = ["pants","shorts","skirt"]


def process_image(image_path):
    """
    Processes the given image and evaluates the response to the given question.
    
    Args:
        image_path (str): Path to the image file.
        question (str): The question string containing the temperature range.
    
    Returns:
        str: The final evaluation result.
    """
    sample_file = PIL.Image.fromarray(image_path)
    image_bytes = sample_file.tobytes()
    sample_file_converted = PIL.Image.frombytes(sample_file.mode, sample_file.size, image_bytes)

    prompt = (
    "Please only provide a nicely formated (in paragraph), brief and complete description on the clothing in this image. Few points to focus on are:\n"

    "- Identify the specific category and type of clothing.\n"
    "- gender suitability (e.g., male, female, unisex).\n"
    "- Overall shape and silhouette.\n"
    "- Material, texture, fabric type and surface features.\n"
    "- Color scheme - Highlight dominant and secondary colors, gradient patterns, and tonal variations.\n"
    "- Describe patterns, designs and their layout .\n"
    "- Suggest appropriateness for events i.e ocassion suitability.\n"
    "- Mention design element details.\n"
    "- Layering and combinations if the item is part of a set or layered with other pieces.\n"
    "- Cultural or regional styles.\n"
    "- Specify suitability for seasons based on material and design.\n"
    "- Functional details that affect practicality.\n"
    "- Complementary accessories visible in the image.\n"
    "- Innovative or unique features highlighting unusual cuts, experimental designs, or standout elements.\n"
    )
    
    response = model_gemini.generate_content([prompt, sample_file_converted])
    return response.text



def detect_clothing(image_path, model, labels, upper, lower):
    """
    Detect clothing in an image, categorize it as 'upper' or 'lower', and return a list of
    detected clothing portions as cropped images along with their labels.

    Parameters:
        image_path (str): Path to the image file.
        model (object): Pretrained model for clothing detection.
        labels (list): List of labels corresponding to class IDs.
        upper (list): List of labels that correspond to upper-body clothing.
        lower (list): List of labels that correspond to lower-body clothing.

    Returns:
        list: A list of dictionaries, each containing:
          - "cropped_image": A numpy array representing the cropped image of the detected clothing.
          - "label": The label of the detected clothing.
          - "category": 'upper', 'lower', or 'other'
    """
    # Run inference
    results = model(image_path)
    boxes = results[0].boxes  # YOLOv8 Boxes object

    # Load the original image
    image = cv2.imread(image_path)
    detected_items = []
    
    # Loop through the detected boxes
    for box in boxes:
        xyxy = box.xyxy[0].cpu().numpy()  # Extract box coordinates
        conf = box.conf[0].cpu().numpy()  # Confidence score
        cls = box.cls[0].cpu().numpy()    # Class ID

        if float(conf) > 0.5:  # Filter detections with confidence > 0.5
            label = labels[int(cls)]

            # Determine the category of the detected item
            if label in upper:
                category = "upper"
            elif label in lower:
                category = "lower"
            else:
                category = "other"
                
            if category != 'other':
                # Extract cropped image
                x1, y1, x2, y2 = map(int, xyxy)
                cropped_image = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
                
                detected_items.append({
                    "cropped_image": cropped_image,
                    "label": label,
                    "category":category
                })
    
    return detected_items