## ✨ Features
- **Accuracy**
  -Dice: 0.9823
  - IoU: 0.9658
  - Precision: 0.9835
  - Recall: 0.9833
  - Accuracy: 0.9922   
  - [Download Checkpoint](https://1024terabox.com/s/1cH2LXa_22BvCWUMaVQJiWw)
 
- **Custom Dataset Loader**
  - Used multiple datasets:
    - [COVID-19 Radiography Database]([https://google.com](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database))
    - [COVID-QU-Ex Dataset]([https://google.com](https://www.kaggle.com/datasets/anasmohammedtahir/covidqu))
    - Total images: 60911
  - Automatically matches images and masks.
  - Handles grayscale conversion, resizing (to 256×256), and CLAHE filtering.

- **Data Augmentation**
  - Horizontal & vertical flips
  - Random rotations (±10°)
  - Gaussian noise injection

- **Model Architecture**
  - U-Net style decoder with skip connections
  - DenseNet121 encoder (ImageNet pretrained)
  - Progressive layer unfreezing for fine-tuning

- **Loss & Metrics**
  - **Loss**: Dice + BCE combined
  - **Metrics**:
    - Dice coefficient
    - IoU (Intersection over Union)
    - Precision
    - Recall
    - **F1-score**
    - **Accuracy**

- **Training Pipeline**
  - Mixed precision training (AMP)
  - Adaptive learning rate with `ReduceLROnPlateau`
  - Checkpoint saving & resuming
  - Early stopping & progressive unfreezing
  - TensorBoard logging



