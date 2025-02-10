# Flower Image Classifier 

## **Project Overview**
This project is an **image classification model** trained to recognize different species of flowers using a deep learning model. It consists of two main scripts:

1. **`train.py`** – Trains a convolutional neural network (CNN) on a dataset of flower images, allowing customization of architecture, hyperparameters, and GPU usage. The trained model is saved as a checkpoint for later use.
2. **`predict.py`** – Loads the trained model to classify images of flowers, returning the top predicted classes and their probabilities. Supports mapping class indices to real flower names and running inference on GPU.

The model uses **transfer learning** with architectures such as **ResNet18** and **VGG13**, optimizing for classification accuracy. The dataset is preprocessed with normalization and augmentation techniques, and training performance is monitored using validation loss and accuracy.

---

## **Installation & Dependencies**
To set up the project, ensure you have **Python 3.7+** and install the required libraries:

```bash
pip install torch torchvision numpy matplotlib argparse PILLOW


## **Usage**

### **Train the Model**
Run `train.py` to train a deep learning model on a dataset of flowers.

```bash
python train.py data_directory
```

### **Make Predictions**
Run `predict.py` to classify a flower image using a trained model.

```bash
python predict.py /path/to/image checkpoint.pth
```


---

## **Project Structure**
```
├── train.py                 # Train the model
├── predict.py               # Predict flower class from an image
├── cat_to_name.json         # JSON mapping of category labels to flower names
├── checkpoint.pth           # Saved model checkpoint
├── README.md                # Project documentation
├── data                     # Dataset directory (not included in the repo)
│   ├── train
│   ├── valid
│   ├── test
```

---

## **Model & Training Details**
- **Pretrained Architectures**: ResNet18, VGG13
- **Loss Function**: Negative Log Likelihood Loss (`NLLLoss`)
- **Optimizer**: Adam
- **Data Augmentation**: Random Resized Cropping, Horizontal Flipping, Normalization
- **Evaluation Metrics**: Validation Loss, Accuracy

---

## **Results**
After training, the model achieves **over 60% accuracy** on the validation dataset, with further improvements possible through hyperparameter tuning and deeper architectures.

---

## **License**
This project is released under the MIT License.
```

---
