# Infant Cry Classification Web App

A web application for classifying infant cries using a machine learning or deep learning model. Users can upload an audio sample of a baby crying, and the app will predict the type of cry along with a confidence score and visualizations.

---

## Requirements

### 🔧 Python Dependencies

Install the required Python packages using pip:

```bash
pip install flask flask-cors numpy librosa matplotlib soundfile scikit-learn tensorflow

⚠️ If you're only using a scikit-learn model (.pkl), you can skip installing tensorflow.

📦 System Dependencies
Some Python packages rely on system-level libraries.

Ubuntu/Debian

bash
Copy
Edit
sudo apt-get install libsndfile1 ffmpeg
macOS

bash
Copy
Edit
brew install libsndfile ffmpeg
Windows Most dependencies should work with pip, but if you run into issues, consider using Anaconda or install precompiled wheels from https://www.lfd.uci.edu/~gohlke/pythonlibs/

Getting Started
1. Run XAMPP
Start Apache using the XAMPP Control Panel.

2. Set Up Project Folder
Place this project folder inside your XAMPP htdocs directory. Example:

bash
Copy
Edit
C:/xampp/htdocs/infant_cry_classification
3. Run the Python Backend
Open a terminal, navigate to the project folder, and run:

bash
Copy
Edit
python app.py
The Flask backend will start on:

arduino
Copy
Edit
http://localhost:8800
4. Open the Web App in Browser
In your browser, go to:

arduino
Copy
Edit
http://localhost/infant_cry_classification/index.php
5. Upload an Audio Sample
Use the form on the page to upload a .wav audio file of a baby cry. The app will return a prediction, confidence level, and visualizations.

Features
Upload .wav audio files for infant cry classification

Real-time audio analysis via Flask API

Works with both scikit-learn and TensorFlow models

Visual output of waveform and spectrogram

Clean API response for integration with other tools

File Structure
bash
Copy
Edit
infant_cry_classification/
├── app.py                      # Flask backend
├── index.php                  # Frontend entry (served by XAMPP)
├── explain.html               # Optional explanation page
├── best_infant_cry_model.pkl  # Trained model (or TensorFlow folder)
├── label_encoder.pkl          # Label encoder
├── uploads/                   # Folder for uploaded audio files
Note: The uploads/ folder will be automatically created if it doesn't exist.

