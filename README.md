# ✍️ Digit Recognizer

The project is based on the [Kaggle Digit Recognizer Database.](https://www.kaggle.com/c/digit-recognizer)

A command-line handwritten digit recognition tool using the MNIST dataset. Designed to recognize digits (0–9) from images with high accuracy (upto 93%).

---

## 📌 Features

- ✅ Train or load a pre-trained ML model (SVM, MLP, or CNN) on MNIST
- ✅ Preprocesses 28×28 grayscale images
- ✅ Quick inference on new images

---

## 🛠️ Tech Stack

- Python
- (Optional) **Machine learning** via scikit-learn or TensorFlow/Keras
- **Image processing** with PIL/OpenCV
- (Optional) **Flask** for web interface

---

## 🔧 Installation

```bash
git clone https://github.com/aravindkumarrr/Digit_Recognizer.git
cd Digit_Recognizer
python3 -m venv venv
source venv/bin/activate        # on macOS/Linux
venv\Scripts\activate           # on Windows
pip install -r requirements.txt
