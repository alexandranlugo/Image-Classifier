#predict.py

#import libraries
import argparse
import torch
import json
import numpy as np
from torchvision import models
from PIL import Image

#command-line arguments
def get_input_args():
    parser = argparse.ArgumentParser(description="Predict the class of an image using a trained model")

    # Required arguments
    parser.add_argument("image_path", type=str, help="Path to input image")
    parser.add_argument("checkpoint", type=str, help="Path to model checkpoint")

    # Optional arguments
    parser.add_argument("--top_k", type=int, default=5, help="Return top K most likely classes")
    parser.add_argument("--category_names", type=str, help="Path to JSON file mapping category labels to names")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for inference")

    return parser.parse_args()

#load checkpoint
def load_checkpoint(filepath):
    """Loads a saved checkpoint and rebuilds the model."""
    checkpoint = torch.load(filepath, map_location="cuda" if torch.cuda.is_available() else "cpu")

    # Load architecture
    arch = checkpoint["arch"]
    if arch == "resnet18":
        model = models.resnet18(pretrained=True)
        in_features = model.fc.in_features
        model.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features, checkpoint["hidden_units"]),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(checkpoint["hidden_units"], 102),
            torch.nn.LogSoftmax(dim=1)
        )
    elif arch == "vgg13":
        model = models.vgg13(pretrained=True)
        in_features = model.classifier[0].in_features
        model.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_features, checkpoint["hidden_units"]),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(checkpoint["hidden_units"], 102),
            torch.nn.LogSoftmax(dim=1)
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.class_to_idx = checkpoint["class_to_idx"]

    return model

#process image
def process_image(image_path):
    """Process an image for use in a PyTorch model."""
    image = Image.open(image_path)

    # Resize keeping aspect ratio
    if image.size[0] < image.size[1]:
        image = image.resize((256, int(256 * image.size[1] / image.size[0])))
    else:
        image = image.resize((int(256 * image.size[0] / image.size[1]), 256))

    # Center crop 224x224
    width, height = image.size
    left = (width - 224) / 2
    top = (height - 224) / 2
    right = left + 224
    bottom = top + 224
    image = image.crop((left, top, right, bottom))

    # Convert to Numpy array
    np_image = np.array(image) / 255.0

    # Normalize channels
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std

    # Reorder dimensions (Color Channel First)
    np_image = np_image.transpose((2, 0, 1))

    return torch.tensor(np_image, dtype=torch.float32)

#make predictions
def predict(image_path, model, top_k, device):
    """Predict the class (or classes) of an image using a trained deep learning model."""
    model.to(device)
    model.eval()

    # Process the image
    image = process_image(image_path)
    image = image.unsqueeze(0).to(device)  # Add batch dimension and move to device

    with torch.no_grad():
        output = model(image)
        probabilities = torch.exp(output)  # Convert log probabilities to probabilities

    # Get the top K probabilities and class indices
    top_p, top_class_indices = probabilities.topk(top_k, dim=1)

    # Convert to numpy
    top_p = top_p.cpu().numpy().squeeze()
    top_class_indices = top_class_indices.cpu().numpy().squeeze()

    # Map class indices to actual class labels
    class_to_idx = model.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}  # Reverse dictionary
    top_classes = [idx_to_class[i] for i in top_class_indices]

    return top_p, top_classes

def main():
    args = get_input_args()

    # Load model
    model = load_checkpoint(args.checkpoint)

    # Use GPU if specified and available
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")

    # Make prediction
    top_p, top_classes = predict(args.image_path, model, args.top_k, device)

    # Map class indices to names if provided
    if args.category_names:
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
        top_classes = [cat_to_name[str(cls)] for cls in top_classes]

    # Display results
    print("\nPredictions:")
    for i in range(args.top_k):
        print(f"{top_classes[i]}: {top_p[i]*100:.2f}%")

if __name__ == "__main__":
    main()
