# Sign Language Recognition System

This project implements a real-time sign language recognition system using deep learning and computer vision.

## Project Structure
```
sign_language_recognition/
├── data/                    # Directory for dataset
├── models/                  # Directory for saved models
├── src/                    # Source code directory
│   ├── data_preprocessing.py
│   ├── model.py
│   ├── train.py
│   └── real_time_prediction.py
├── requirements.txt
└── README.md
```

## Setup Instructions

1. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

2. Download the dataset:
   - Go to Kaggle: https://www.kaggle.com/datasets/ardamavi/sign-language-digits-dataset
   - Click "Download" button
   - Extract the downloaded zip file into the `data` directory

3. Run the training:
   ```bash
   python src/train.py
   ```

4. Run real-time prediction:
   ```bash
   python src/real_time_prediction.py
   ```

## Dataset Information
The dataset contains images of hand gestures representing digits 0-9 in sign language. Each image is 100x100 pixels in size. 