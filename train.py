#train.py

#import libraries
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import os

#define command-line arguments
def get_input_args():
    parser = argparse.ArgumentParser(description="Train a deep learning model on a dataset")

    # Required argument: dataset directory
    parser.add_argument("data_dir", type=str, help="Path to dataset directory")

    # Optional arguments
    parser.add_argument("--save_dir", type=str, default=".", help="Directory to save checkpoint")
    parser.add_argument("--arch", type=str, default="resnet18", choices=["resnet18", "vgg13"], help="Model architecture")
    parser.add_argument("--learning_rate", type=float, default=0.003, help="Learning rate")
    parser.add_argument("--hidden_units", type=int, default=512, help="Number of hidden units")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for training")

    return parser.parse_args()

#load and process data
def load_data(data_dir):
    train_dir = os.path.join(data_dir, "train")
    valid_dir = os.path.join(data_dir, "valid")

    # Define data transformations
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    valid_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load datasets
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_transforms)

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=32, shuffle=False)

    return train_loader, valid_loader, train_dataset.class_to_idx

#build the model
def build_model(arch, hidden_units):
    if arch == "resnet18":
        model = models.resnet18(pretrained=True)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(in_features, hidden_units),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_units, 102),  # Output layer
            nn.LogSoftmax(dim=1)
        )
    elif arch == "vgg13":
        model = models.vgg13(pretrained=True)
        in_features = model.classifier[0].in_features
        model.classifier = nn.Sequential(
            nn.Linear(in_features, hidden_units),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_units, 102),
            nn.LogSoftmax(dim=1)
        )

    # Freeze pretrained parameters
    for param in model.parameters():
        param.requires_grad = False

    return model

#train model
def train_model(model, train_loader, valid_loader, device, criterion, optimizer, epochs):
    model.to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            logps = model(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Validation step
        model.eval()
        valid_loss = 0
        accuracy = 0

        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                logps = model(inputs)
                batch_loss = criterion(logps, labels)
                valid_loss += batch_loss.item()

                ps = torch.exp(logps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        print(f"Epoch {epoch+1}/{epochs}.. "
              f"Train loss: {running_loss/len(train_loader):.3f}.. "
              f"Validation loss: {valid_loss/len(valid_loader):.3f}.. "
              f"Validation accuracy: {accuracy/len(valid_loader)*100:.2f}%")

#save model
def save_checkpoint(model, save_dir, class_to_idx, arch, hidden_units):
    checkpoint = {
        "arch": arch,
        "hidden_units": hidden_units,
        "model_state_dict": model.state_dict(),
        "class_to_idx": class_to_idx
    }
    torch.save(checkpoint, os.path.join(save_dir, "checkpoint.pth"))
    print("Model checkpoint saved!")

def main():
    args = get_input_args()

    # Use GPU if specified and available
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")

    # Load data
    train_loader, valid_loader, class_to_idx = load_data(args.data_dir)

    # Build model
    model = build_model(args.arch, args.hidden_units)

    # Define criterion and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=args.learning_rate)

    # Train the model
    train_model(model, train_loader, valid_loader, device, criterion, optimizer, args.epochs)

    # Save checkpoint
    save_checkpoint(model, args.save_dir, class_to_idx, args.arch, args.hidden_units)

if __name__ == "__main__":
    main()


