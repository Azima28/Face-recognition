# VGG16 Gender Classification

Gender classification system using VGG16 deep learning model on the CelebA dataset. This project implements transfer learning to classify facial images as male or female with high accuracy.

## 📋 Features

- **Duplicate Image Detection**: Automated detection and removal of duplicate images using perceptual hashing
- **Data Augmentation**: Random flips, color jitter, and rotation for robust training
- **Transfer Learning**: Pre-trained VGG16 with frozen feature layers and fine-tuned classifier
- **Comprehensive Evaluation**: Classification reports, confusion matrix, and accuracy metrics
- **Visualization**: Training curves and sample predictions

## 🎯 Performance

- **Test Accuracy**: 92.1%
- **Training Time**: ~650 seconds (15 epochs)
- **Inference Speed**: ~0.000038 seconds per image
- **Best Test Accuracy**: 92.8% (Epoch 14)

### Classification Metrics

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Female (0) | 0.95 | 0.92 | 0.93 | 591 |
| Male (1) | 0.88 | 0.93 | 0.91 | 409 |

## 🛠️ Requirements

```bash
pip install torch torchvision
pip install pandas numpy matplotlib seaborn
pip install scikit-learn
pip install pillow
pip install imagehash
```

## 📁 Dataset Structure

```
FaceRecognition/
├── Images/
│   ├── 000001.jpg
│   ├── 000002.jpg
│   └── ...
├── list_attribute.txt
└── class_identity.txt
```

## 🚀 Usage

### 1. Mount Google Drive (for Colab)

```python
from google.colab import drive
drive.mount('/content/drive')
```

### 2. Data Preparation

```python
data_path = "/content/drive/MyDrive/Colab/FaceRecognition"
image_path = os.path.join(data_path, "Images")
```

### 3. Remove Duplicates

```python
remover = DuplicateRemover(image_path, hash_size=18)
remover.find_duplicates()
```

### 4. Train Model

```python
# Configuration
batch_size = 64
learning_rate = 0.001
epochs = 15

# Initialize model
model = models.vgg16(weights="IMAGENET1K_V1")
model.classifier[6] = nn.Linear(4096, 2)

# Train
for epoch in range(epochs):
    # Training loop
    ...
```

### 5. Evaluate

```python
# Generate predictions
model.eval()
with torch.no_grad():
    for data in test_loader:
        # Evaluation loop
        ...

# Print classification report
print(classification_report(y_true, y_pred))
```

## 📊 Model Architecture

**VGG16 with Transfer Learning:**
- **Feature Extractor**: Frozen pre-trained VGG16 layers (ImageNet weights)
- **Classifier**: Fine-tuned last 3 layers
- **Output**: 2 classes (Female/Male)
- **Optimizer**: SGD with momentum (0.9)
- **Loss Function**: CrossEntropyLoss

## 🔧 Data Augmentation

```python
transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
```

## 📈 Training Progress

| Epoch | Train Loss | Test Loss | Train Acc | Test Acc |
|-------|------------|-----------|-----------|----------|
| 1 | 0.3617 | 0.2748 | 83.45% | 89.20% |
| 5 | 0.1982 | 0.2072 | 91.97% | 91.20% |
| 10 | 0.1710 | 0.1971 | 92.88% | 91.40% |
| 15 | 0.1359 | 0.1853 | 94.67% | 91.90% |

## 🎨 Visualizations

The project includes:
- Gender distribution bar plot
- Sample image grid with predictions
- Training vs Test Loss curves
- Training vs Test Accuracy curves
- Confusion matrix heatmap

## 📝 Dataset Information

- **Total Images**: 5,000 (after duplicate removal)
- **Training Set**: 4,000 images (80%)
- **Test Set**: 1,000 images (20%)
- **Female**: 2,953 images (59.1%)
- **Male**: 2,047 images (40.9%)

## 🔍 Key Components

### DuplicateRemover Class
Detects and removes duplicate images using average hash algorithm with configurable hash size.

### FaceDataset Class
Custom PyTorch Dataset for loading and preprocessing facial images with labels.

### Model Training
- Stratified train-test split
- Batch processing with DataLoader
- Model checkpointing (saves best model)
- Real-time accuracy monitoring

## 💾 Model Saving

Best performing model is automatically saved as `best_vgg16.pth` based on test accuracy.

```python
if test_accuracy > best_acc:
    best_acc = test_accuracy
    torch.save(model.state_dict(), "best_vgg16.pth")
```

## 🖥️ Hardware Requirements

- **GPU**: CUDA-compatible GPU recommended
- **RAM**: Minimum 8GB
- **Storage**: ~2GB for dataset and models

## 📄 License

This project is available for educational and research purposes.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or feedback, please open an issue in the repository.

---

**Note**: This project uses the CelebA dataset. Please ensure you have the appropriate rights and permissions to use the dataset for your purposes.
