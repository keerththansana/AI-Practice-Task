# AI Project Collection

This repository contains a collection of artificial intelligence and machine learning projects implemented in Python. Each project demonstrates different AI techniques and applications.


## 📁 Project Structure

├── Image_Captioning.py
├── Recommendation_System.py
├── simple_chatbot.py
├── TIC_TAC_TOE_AI.py
├── FACE_DETECTION_AND_RECOGNITION.py
└── README.md


## 🚀 Projects Overview

### 1. Image Captioning
**File:** `Image_Captioning.py`

A deep learning model that generates descriptive captions for images using a combination of CNN (VGG16) for feature extraction and LSTM for sequence generation.

**Features:**
- Pre-trained VGG16 for image feature extraction
- LSTM-based decoder for caption generation
- Tokenization and sequence processing
- Training pipeline for custom datasets

**Requirements:**
- TensorFlow/Keras
- NumPy
- Pillow

### 2. Recommendation System
**File:** `Recommendation_System.py`

A collaborative filtering recommendation system that suggests items to users based on cosine similarity between user preferences.

**Features:**
- User-item matrix implementation
- Cosine similarity calculations
- Personalized recommendations
- Pandas for data manipulation

**Requirements:**
- pandas
- scikit-learn

### 3. Simple Chatbot
**File:** `simple_chatbot.py`

A rule-based chatbot that responds to user inputs using pattern matching and predefined responses.

**Features:**
- Case-insensitive input matching
- Multiple conversation scenarios
- Interactive command-line interface
- Easy-to-extend response rules

**Requirements:**
- No external dependencies (pure Python)

### 4. Tic-Tac-Toe AI
**File:** `TIC_TAC_TOE_AI.py`

An AI player for Tic-Tac-Toe using the Alpha-Beta pruning algorithm for optimal move selection.

**Features:**
- Alpha-Beta pruning implementation
- Game state evaluation
- Minimax algorithm optimization
- Efficient move searching

**Note:** This file contains the core Alpha-Beta algorithm. You'll need to implement the game logic (`game_over`, `evaluate`, `get_available_moves`, `make_move`) to complete the game.

### 5. Face Detection and Recognition
**File:** `FACE_DETECTION_AND_RECOGNITION.py`

A computer vision application that detects faces in images/video and performs face recognition.

**Features:**
- Haar cascades for face detection
- Integration with deep learning models for recognition
- Real-time processing capabilities
- OpenCV for image processing

**Requirements:**
- OpenCV
- dlib
- TensorFlow/Keras
- NumPy
- imutils

## 🛠 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd ai-projects-collection
   ```

2. **Install required dependencies:**
    ```bash
   pip install tensorflow opencv-python dlib imutils scikit-learn pandas numpy pillow
    ```

3. **Run individual projects:**
  ```bash
   python simple_chatbot.py
   python Recommendation_System.py
   # Note: Some projects require additional setup (see below)
   ```

## 📋 Prerequisites
    Python 3.7+
    TensorFlow 2.x
    OpenCV
    scikit-learn
    pandas
    NumPy

## 🎯 Usage Examples
### Chatbot
```bash
python simple_chatbot.py
# Type your messages and type "exit" to quit
```

### Recommendation System
```bash
# Get recommendations for a user
recommendations = get_recommendations('User1')
print(recommendations)
```

## ⚠️ Important Notes
Some projects require additional setup:
Image Captioning:
Pre-trained VGG16 weights will be downloaded automatically
You need to provide your own image dataset and captions
Face Recognition:
You need to provide your own trained model at 'path_to_face_recognition_model'
Requires webcam or video input for real-time detection
Tic-Tac-Toe:
Complete the game logic functions (game_over, evaluate, get_available_moves, make_move) to make it fully functional

## 🔧 Customization
Each project can be customized:
Image Captioning: Train on your own dataset of images and captions
Recommendation System: Modify the user-item matrix with your data
Chatbot: Add more response rules and patterns
Face Recognition: Train with your own face dataset

## 🤝 Contributing
Feel free to contribute to this project by:
Adding new AI projects
Improving existing implementations
Fixing bugs
Adding documentation

## 📝 License
This project is open source and available under the MIT License.




