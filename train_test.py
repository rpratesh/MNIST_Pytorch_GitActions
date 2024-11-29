import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from model import Model



def train(model, train_loader, epochs, optimizer, criterion, device):
    model.train()
    for epoch in range(epochs):
        train_loss = 0
        correct = 0
        total_samples = 0  # Keep track of total samples processed
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            correct += output.argmax(dim=1).eq(target).sum().item()
            total_samples += target.size(0)  # Add batch size to total
        
        # Calculate average loss and accuracy for the epoch
        avg_train_loss = train_loss/len(train_loader)
        train_acc = 100. * correct / total_samples
        
        print(f"Epoch: {epoch+1}")
        print(f"Average training Loss: {avg_train_loss:.4f}")
        print(f"Training Accuracy: {train_acc:.4f}")

        return avg_train_loss, train_acc


def test(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    test_acc = 0
    correct = 0
    preds_total = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            # print(target.size())
            preds = output.argmax(dim=1)
            preds_total += len(preds)
            correct += target.eq(preds).sum().item()
            test_loss += criterion(output, target)
            # print(test_loss)
    test_acc = 100. * correct / preds_total
    avg_test_loss = (test_loss/ len(test_loader)).item()  #len of test loader gives total number of batches.
    print(f"Average test loss: {avg_test_loss:.4f}")
    print("Test Accuracy:",test_acc)
    return avg_test_loss, test_acc

def infer_image(model, image_path, ground_truth):
    #Read image


    #Load image into model

    #Print the predicted label and ground_truth
    
    pass

def create_dataset_loaders(augmentation=True, save_local=False):
    transform_aug = transforms.Compose([
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 10% translation
        transforms.RandomRotation(10),      # Randomly rotate the image by 10 degrees
        transforms.ToTensor(),              # Convert the image to a tensor
        transforms.Normalize((0.1307,), (0.3081,))  # Normalize the image
    ])
    # Define transforms with normalization
    transform_noaug = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])

    # Create MNIST dataset loader with normalization
    if augmentation:
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform_aug)
        if save_local:
            # Create a directory to save augmented images
            save_dir = './augmented_images'
            os.makedirs(save_dir, exist_ok=True)

            # Save a few augmented images (e.g., first 10 images in the dataset)
            for i in range(64):
                # Get the image and label from the dataset
                image, label = train_dataset[i]
                
                # Save the image
                save_image(image, os.path.join(save_dir, f'augmented_image_{i}.png'))

            print(f'Saved augmented images to {save_dir}')
    else:
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform_noaug)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform_noaug)
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

    return train_loader,test_loader 


def count_parameters(model):
    """
    Calculate the total number of trainable parameters in the model
    
    Args:
        model: PyTorch model
        
    Returns:
        total_params: Total number of trainable parameters
        trainable_params: Number of trainable parameters
        non_trainable_params: Number of non-trainable parameters
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': non_trainable_params
    }

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.CrossEntropyLoss()
    nn_model = Model().to(device)
    
    # Print model parameters before training
    params = count_parameters(nn_model)
    print(f"Total parameters: {params['total']:,}")
    
    # Use Adam optimizer with a higher learning rate for faster convergence
    optimizer = optim.Adam(nn_model.parameters(), lr=0.003)

    train_loader,test_loader = create_dataset_loaders(False, False)
    
    # Train for 1 epoch
    train(nn_model, train_loader, 1, optimizer, criterion, device)
    test(nn_model, test_loader, criterion, device)

