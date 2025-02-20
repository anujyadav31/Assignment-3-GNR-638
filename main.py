import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import cv2

# --- 1. Dataset Preparation ---
# Update the path below to where your UC Merced dataset is stored.
data_dir = 'data'

# Define transforms (we assume images are 256x256 already, but you might normalize, etc.)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    # Optionally, add normalization:
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Use ImageFolder to load the dataset (ensure folder structure: data_dir/class_name/image.jpg)
dataset = ImageFolder(root=data_dir, transform=transform)

# Split dataset into training and validation (for example: 80/20 split)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)


# --- 2. Define a Small CNN Model ---
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=21):
        super(SimpleCNN, self).__init__()
        # Feature extractor: three conv blocks
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),  # 256x256 -> 256x256
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 256x256 -> 128x128

            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # 128x128 -> 128x128
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 128x128 -> 64x64

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 64x64 -> 64x64
            nn.ReLU(inplace=True)
            # Note: We do not pool here to keep spatial resolution for CAM.
        )
        # Global Average Pooling and final classifier
        self.gap = nn.AdaptiveAvgPool2d((1, 1))  # outputs a 1x1 feature per channel
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.features(x)
        # Save feature maps for CAM extraction
        self.feature_maps = x  # Shape: [batch_size, 64, H, W]
        x = self.gap(x)  # Shape: [batch_size, 64, 1, 1]
        x = x.view(x.size(0), -1)  # Flatten to [batch_size, 64]
        out = self.fc(x)  # [batch_size, num_classes]
        return out


# Instantiate the model, loss, and optimizer.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN(num_classes=21).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --- 3. Training the Model ---
num_epochs = 10  # Adjust epochs as needed

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    # Optionally add validation here

print("Training complete.")


# --- 4. Generate Class Activation Maps (CAM) ---

def generate_cam(feature_maps, fc_weights, class_idx):
    """
    Compute the CAM for a given class.
    :param feature_maps: Tensor of shape [1, C, H, W]
    :param fc_weights: Weight matrix of shape [num_classes, C] from the fc layer
    :param class_idx: The target class index
    :return: A CAM image (numpy array) resized to the original image size.
    """
    # Multiply each feature map by its corresponding weight for the target class and sum
    # feature_maps: [1, C, H, W] --> remove batch dim
    feature_maps = feature_maps.squeeze(0)  # now [C, H, W]
    cam = torch.zeros(feature_maps.shape[1:], dtype=torch.float32).to(device)
    for i in range(fc_weights.shape[1]):
        cam += fc_weights[class_idx, i] * feature_maps[i, :, :]

    # Apply ReLU and normalize between 0 and 1
    cam = torch.relu(cam)
    cam = cam - cam.min()
    if cam.max() != 0:
        cam = cam / cam.max()
    cam = cam.detach().cpu().numpy()

    # Resize CAM to original image size (256x256)
    cam = cv2.resize(cam, (256, 256))
    return cam


def overlay_cam_on_image(image, cam, alpha=0.5):
    """
    Overlay the CAM on the original image.
    :param image: Original image as a numpy array (H, W, 3) in RGB format.
    :param cam: CAM as a grayscale numpy array (H, W) with values in [0, 1].
    :param alpha: Weight for the heatmap overlay.
    :return: Overlay image.
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = heatmap * alpha + image * (1 - alpha)
    return np.uint8(overlay)


# --- 5. Visualize CAM for 1-2 Images ---
# Let’s take one image from the validation set
model.eval()
# Get a sample (no shuffling in val_loader ensures reproducibility)
sample_img, sample_label = val_dataset[0]
input_tensor = sample_img.unsqueeze(0).to(device)  # Add batch dimension

# Forward pass
output = model(input_tensor)
# Predicted class
pred_class = output.argmax(dim=1).item()
print(f"Predicted class: {pred_class}, True class: {sample_label}")

# Get the weights from the fc layer
# fc layer weight shape: [num_classes, 64]
fc_weights = model.fc.weight.data

# Generate CAM from the stored feature maps
cam = generate_cam(model.feature_maps, fc_weights, pred_class)

# To overlay, we need the original image in a proper (H, W, 3) format and in [0,255]
# Undo the normalization (assuming standard ImageNet stats used above)
inv_normalize = transforms.Normalize(
    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
    std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
)
# Convert tensor back to PIL Image for visualization
un_tensor = inv_normalize(sample_img)
# Clamp values to [0,1]
un_tensor = torch.clamp(un_tensor, 0, 1)
# Convert to numpy array in HxWxC format
orig_img = un_tensor.permute(1, 2, 0).detach().cpu().numpy()
orig_img = np.uint8(255 * orig_img)

# Overlay CAM on the original image
overlay_img = overlay_cam_on_image(orig_img, cam, alpha=0.5)

# Display the results
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(orig_img)
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(cam, cmap='jet')
plt.title("Class Activation Map")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(overlay_img)
plt.title("Overlay")
plt.axis('off')
plt.show()
