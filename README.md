# README

# 1. Downloading and Creating Fashion Dataset

This script is designed to convert object detection datasets (e.g., Fashionpedia) into the YOLO format. It processes images and annotations, saving them in the required directory structure and format compatible with YOLO.

### Features
1. **Image Processing**: Saves dataset images into a specified directory (`datasets/fashion_dataset/<split>/images`).
2. **Annotation Conversion**: Converts bounding box annotations from Pascal VOC format to YOLO format and saves them as `.txt` files in a corresponding labels directory (`datasets/fashion_dataset/<split>/labels`).
3. **Directory Setup**: Ensures the required YOLO directory structure (`datasets`, `images`, `labels`) is created automatically.

### File Structure
After execution, the processed dataset will follow this structure:
```
datasets/
└── fashion_dataset/
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── validation/
    │   ├── images/
    │   └── labels/
    └── test/
        ├── images/
        └── labels/
```

### Requirements
- **Libraries**: `numpy`, `pandas`, `datasets`, `os`, `tqdm`.
- **Dataset**: The script uses `detection-datasets/fashionpedia` as an example. Ensure the dataset is available and loaded properly.

# 2. Clothing Detection Fine Tuning 

This script fine-tunes a YOLO model for a custom object detection dataset.

### Key Features
- **Dataset Configuration**: 
  - Specifies paths to training and validation images.
  - Defines 46 classes with detailed category names.
  - Saves configuration as a `data.yaml` file.

- **Model Training**: 
  - Fine-tunes a pretrained YOLO model (`yolo11n.pt`) for 5 epochs with a batch size of 32.
  - Freezes the first 10 layers during training for efficient fine-tuning.

- **Output**: 
  - Saves the fine-tuned model as `Yolo11n_Finetuned.pt`.

### Requirements
- Libraries: `ultralytics`, `yaml`.
- Dataset: Images structured for YOLO format with corresponding annotations.

### Usage
1. Modify `train` and `val` paths in `data_config` for your dataset.
2. Run the script to fine-tune the YOLO model.
3. The fine-tuned model will be saved locally for future use.


# 3. Prompting for Image Description (Vision Language Model)  

## NOTE : Due to resource and time complexity description of only 3000 images could be generated.

The pipeline in this file is for detecting clothing in images and generating detailed descriptions for each detected item. It combines a fine-tuned YOLO model for object detection with generative AI (Gemini) as a vision language model.

## Features

1. **Clothing Detection**:  
   The fine-tuned YOLO model (`Yolo11n_Finetuned.pt`) detects clothing items in images with labels like shirts, pants, dresses, etc. Each detection is categorized as `upper`, `lower`, or `other`.

2. **Description Generation**:  
   Using Google Gemini's generative AI model (`gemini-2.0-flash-exp`), detailed descriptions are created for 'upper' and 'lower' clothing only for now. The descriptions include:
   - Category and type of clothing
   - Gender suitability
   - Material, texture, and fabric details
   - Color scheme and patterns
   - Event and season appropriateness
   - Design elements and cultural styles
   - Practical and functional details

3. **Pipeline Automation**:  
   The script processes a directory of images in batches, handling errors gracefully (e.g., API rate limits) and saving results incrementally.

4. **Output**:  
   The detected items and their descriptions are saved in a structured JSON file (`Cloth Description.json`).

## Requirements

- **Python Packages**:
  - `ultralytics` for YOLO
  - `opencv-python` for image processing
  - `matplotlib` and `Pillow` for image manipulation
  - `google.generativeai` for integrating Gemini
  - `tqdm` for progress tracking
  - `json` for saving output
- **Models**:
  - YOLO model: `Yolo11n_Finetuned.pt`
  - Google Gemini: API key required for generative model usage

## Output Format

The output JSON file contains data for each processed image, structured as:

```json
{
  "image_path": {
    "upper": "Description of upper clothing item",
    "lower": "Description of lower clothing item"
  }
}
```

# 4. CLIP Training

## NOTE : Due to compute and time complexity training on a small sample of only 3000 image-text pairs could be done.

A custom CLIP-based multi-modal model has been trained for embedding and similarity search.

## Key Features

1. **Clothing Detection**:
   - Detects and classifies clothing items (upper and lower garments) in images.
   - Leverages a fine-tuned YOLO model (`Yolo11n_Finetuned.pt`) for object detection.
   - Supports label filtering and categorization for upper garments, e.g., shirts, jackets, and dresses.

2. **Image Cropping and Metadata**:
   - Crops detected clothing items from input images.
   - Associates cropped images with descriptions stored in a `cloth_description.json` file.
   - Processes only upper garments due to computational constraints.

3. **CLIP-Based Recommendation**:
   - Implements a multi-modal CLIP model to align image and text embeddings.
   - Uses ResNet50 for image encoding and a transformer-based text encoder (`Alibaba-NLP/gte-large-en-v1.5`).
   - Projects embeddings to a common space using learned projection layers.

4. **Dataset Handling**:
   - Creates a dataset of image-text pairs for training and evaluation.
   - Splits data into training and test sets with support for transformations.

5. **Training and Evaluation**:
   - Trains the CLIP model using cross-entropy loss to align embeddings.
   - Supports evaluation and visualization of training and test losses.

6. **Image Embedding Generation**:
   - Extracts image embeddings for use in similarity search and recommendation tasks.
   - Saves the generated embeddings for further processing.

## Pipeline Overview

1. **Detection**:
   - Use YOLO to detect clothing items in images.
   - Filter and crop images for upper garments.
   - Save cropped images and their descriptions.

2. **CLIP Training**:
   - Prepare a dataset of image-text pairs.
   - Train the CLIP model to align image and text embeddings.

3. **Embedding Generation**:
   - Generate embeddings for images using the trained CLIP model.
   - Save embeddings for downstream tasks, such as recommendation.

4. **Recommendation**:
   - Use cosine similarity in the embedding space for image similarity search and clothing recommendations.

## File Structure

- `Yolo11n_Finetuned.pt`: Fine-tuned YOLO model for clothing detection.
- `Cloth Description.json`: Metadata with descriptions of clothing items.
- `upper_garment_description.json`: Output file with descriptions of detected upper garments.
- `clip.pt`: Saved weights of the trained CLIP model.
- `image_embeddings.pt`: Generated embeddings for upper garment images.


# 5. Image Similarity Search with CLIP Inference

This repository demonstrates an image similarity search pipeline using a Vision-Language Model (VLM) with CLIP for embedding and inference. It matches textual descriptions with the most relevant images in a dataset.

## Features

1. **Text-to-Image Retrieval**:
   - Given a textual query, the system retrieves the top `n` images from a pre-embedded image dataset based on cosine similarity.

2. **Pipeline Highlights**:
   - **Textual Description Generation**: Extracts textual descriptions from input images using a YOLO-based clothing detection model and a Gemini Flash Multimodal model.
   - **Dimensionality Reduction**: Embeddings are processed through a pretrained sentence transformer and CLIP model to project them into a lower dimension for similarity computation.

3. **Visualization**:
   - Displays the top matching images and their captions using Matplotlib.

## Components

### Key Functions
1. **`inference`**:
   - Accepts a query and retrieves the top `n` relevant images based on similarity.
   - Visualizes the results in a grid format.

2. **`vlm_text`**:
   - Processes an input image to generate a textual description using a YOLO-based clothing detection model.

### Input Data
- **Image Dataset**: Stored in a JSON file (`Cloth Description.json`) with paths and corresponding captions.
- **Image Embeddings**: Precomputed and stored in `image_embeddings.pt`.
- **Model Checkpoint**: CLIP model weights loaded from `clip.pt`.

### Inference Workflow
1. Generate a textual description of an image (`vlm_text`).
2. Use the textual query in the `inference` function to retrieve and visualize the most relevant images from the dataset.

### Prerequisites
- Python 3.8+
- Required libraries: `torch`, `timm`, `transformers`, `opencv-python`, `Pillow`, `pandas`
