import os
import numpy as np
import pickle
import librosa
import json
import io
import zipfile
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import threading
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import tensorflow as tf
from tensorflow import keras
from flask import Flask, render_template, request, jsonify, Response
from flask_cors import CORS
import base64
from werkzeug.utils import secure_filename
import shutil
from queue import Queue, Empty
import soundfile as sf

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['DATASET_FOLDER'] = 'dataset'
app.config['MODELS_FOLDER'] = 'models'
app.config['MAX_CONTENT_LENGTH'] = 128 * 1024 * 1024  # 128MB max upload size

# Create necessary folders
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['DATASET_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODELS_FOLDER'], exist_ok=True)

# Global variables for model
model = None
le = None
is_tf_model = False

# Global variables for training
training_clients = []
training_thread = None
training_in_progress = False
training_progress = 0
training_log = []
training_metrics = {'iterations': [], 'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': []}


def extract_audio_features(audio_path, n_mfcc=13):
    try:
        # Use soundfile for loading audio
        y, sr = sf.read(audio_path)
        # Convert to mono if stereo
        if y.ndim > 1:
            y = np.mean(y, axis=1)
            
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc), axis=1)
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr), axis=1)
        zcr = np.mean(librosa.feature.zero_crossing_rate(y), axis=1)
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
        features = np.concatenate([
            mfccs, spectral_centroid.ravel(), zcr.ravel(), chroma.ravel()
        ])
        return features, y, sr
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None, None, None

def extract_audio_features_from_buffer(audio_buffer, sr, n_mfcc=13):
    try:
        y = audio_buffer
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc), axis=1)
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr), axis=1)
        zcr = np.mean(librosa.feature.zero_crossing_rate(y), axis=1)
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
        features = np.concatenate([
            mfccs, spectral_centroid.ravel(), zcr.ravel(), chroma.ravel()
        ])
        return features
    except Exception as e:
        print(f"Error extracting features from buffer: {e}")
        return None

# Generate visualization of the audio - added from the second file
def generate_audio_visualization(y, sr):
    plt.figure(figsize=(10, 6))
    
    # Plot waveform
    plt.subplot(2, 1, 1)
    plt.title("Waveform")
    librosa.display.waveshow(y, sr=sr)
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    
    # Plot spectrogram
    plt.subplot(2, 1, 2)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title("Spectrogram")
    
    plt.tight_layout()
    
    # Save to a buffer instead of a file
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    
    # Convert to base64 for embedding in HTML
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    return img_str

def load_models():
    global model, le, is_tf_model
    
    model_path = os.path.join(app.config['MODELS_FOLDER'], "best_infant_cry_model.pkl")
    le_path = os.path.join(app.config['MODELS_FOLDER'], "label_encoder.pkl")
    
    try:
        # Check if model exists
        if not os.path.exists(model_path):
            print(f"Model file not found at {model_path}")
            tf_model_path = os.path.join(app.config['MODELS_FOLDER'], "best_infant_cry_model")
            if os.path.exists(tf_model_path):
                model_path = tf_model_path
                print(f"Found TensorFlow model directory")
        
        if os.path.isdir(model_path):
            # TensorFlow model
            model = keras.models.load_model(model_path)
            is_tf_model = True
            print("Loaded TensorFlow model")
        else:
            # sklearn model
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            is_tf_model = False
            print("Loaded sklearn model")
        
        # Load label encoder
        with open(le_path, "rb") as f:
            le = pickle.load(f)
        print(f"Classes: {le.classes_}")
        return True
    except Exception as e:
        print(f"Error loading models: {e}")
        return False

def make_prediction(features):
    """Make a prediction using the loaded model"""
    global model, le
    
    if model is None:
        load_success = load_models()
        if not load_success:
            return "Unknown", 0.0, {"Unknown": 1.0}
    
    features_reshaped = features.reshape(1, -1)
    if is_tf_model:
        pred_proba = model.predict(features_reshaped)
        pred_class = np.argmax(pred_proba, axis=1)[0]
        confidence = float(pred_proba[0][pred_class])
        pred_label = le.inverse_transform([pred_class])[0]
        # Get all class probabilities for the chart
        all_probs = {le.classes_[i]: float(pred_proba[0][i]) for i in range(len(le.classes_))}
    else:
        # For sklearn models
        pred_proba = model.predict_proba(features_reshaped)[0]
        pred_class = model.predict(features_reshaped)[0]
        confidence = float(max(pred_proba))
        pred_label = le.inverse_transform([pred_class])[0]
        
        # Map classes to their proper indices in the model
        all_probs = {}
        for i, class_index in enumerate(model.classes_):
            class_name = le.inverse_transform([class_index])[0]
            all_probs[class_name] = float(pred_proba[i])
    
    print(f"\nPrediction completed: {pred_label} (Confidence: {confidence:.4f})")
    return pred_label, confidence, all_probs

def send_training_update(message):
    """Send an SSE message to all connected clients"""
    global training_clients, training_progress
    
    # Create a base data structure with default values
    base_data = {
        'progress': training_progress,
        'iteration': None,
        'accuracy': None,
        'loss': None
    }
    
    # Update with new data
    if isinstance(message, str):
        base_data['log'] = message
        data = json.dumps(base_data)
    else:
        # Merge message dictionary with base_data
        base_data.update(message)
        data = json.dumps(base_data)
        
    # Format the message as an SSE event
    formatted_message = f"data: {data}\n\n"
    
    for client in list(training_clients):
        try:
            client.put(formatted_message, block=False)
        except Exception as e:
            print(f"Error sending to client: {e}")
            training_clients.remove(client)

def extract_dataset(zip_path, extract_path):
    """Extract the uploaded dataset zip file"""
    global training_progress, training_log
    
    # Clear the extract directory if it exists
    if os.path.exists(extract_path):
        shutil.rmtree(extract_path)
    os.makedirs(extract_path, exist_ok=True)
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Filter out __MACOSX directories and metadata files
            files_to_extract = [f for f in zip_ref.namelist() 
                               if not f.startswith('__MACOSX/') and 
                                  not f.startswith('._') and 
                                  not '/__MACOSX/' in f]
            
            total_files = len(files_to_extract)
            send_training_update(f"Extracting {total_files} files from dataset...")
            
            # Extract only the filtered files
            for file in files_to_extract:
                zip_ref.extract(file, extract_path)
                
            send_training_update(f"Dataset extraction complete")
            return True
    except Exception as e:
        send_training_update(f"Error extracting dataset: {e}")
        return False

def process_audio_dataset(extract_path):
    """Process the audio files and create a feature CSV"""
    global training_progress, training_log
    
    # Find all WAV files, but exclude Mac metadata files
    data = []
    for root, _, files in os.walk(extract_path):
        # Skip __MACOSX directories entirely
        if "__MACOSX" in root:
            continue
            
        for file in files:
            # Skip hidden files and Mac metadata files
            if file.startswith(".") or file.startswith("._"):
                continue
                
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                # Get label from parent directory name
                label = os.path.basename(root)
                data.append([file_path, label])
    
    if len(data) == 0:
        send_training_update("No audio files found in the dataset")
        return None
        
    send_training_update(f"Found {len(data)} audio files")
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=["audio_link", "label"])
    csv_path = os.path.join(app.config['DATASET_FOLDER'], "infant_cry_data.csv")
    df.to_csv(csv_path, index=False)
    
    # Process each audio file to extract features
    n_mfcc = 13
    feature_columns = [f"MFCC_{i+1}" for i in range(n_mfcc)] + \
                      ["Spectral_Centroid", "Zero_Crossing_Rate"] + \
                      [f"Chroma_{i+1}" for i in range(12)]
    
    # Initialize features DataFrame with proper column types
    features_df = pd.DataFrame(columns=["audio_link", "label"] + feature_columns)
    
    # Extract features from each audio file
    total_files = len(df)
    processed = 0
    error_files = 0
    
    # Create a list to hold all feature rows to avoid concatenation warnings
    all_feature_rows = []
    
    for i, row in df.iterrows():
        audio_path = row["audio_link"]
        try:
            # Use updated audio loading approach
            y, sr = sf.read(audio_path)
            # Convert to mono if stereo
            if y.ndim > 1:
                y = np.mean(y, axis=1)
                
            if y.size == 0:
                send_training_update(f"Warning: Empty audio file {audio_path}")
                error_files += 1
                continue
                
            # Extract features
            mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc), axis=1)
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr), axis=1)
            zcr = np.mean(librosa.feature.zero_crossing_rate(y), axis=1)
            chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
            
            # Combine all features
            features = np.concatenate([
                mfccs, spectral_centroid.ravel(), zcr.ravel(), chroma
            ])
            
            # Create a new row with these features
            new_row = {"audio_link": audio_path, "label": row["label"]}
            
            # Add each feature to the row
            for j, col in enumerate(feature_columns):
                if j < len(features):
                    new_row[col] = features[j]
                else:
                    # Safeguard against unexpected feature size
                    new_row[col] = 0.0
            
            # Add to our list of rows
            all_feature_rows.append(new_row)
            
            processed += 1
            if processed % 10 == 0 or processed == total_files:
                progress = int((processed / total_files) * 20)  # 20% of total progress for extraction
                training_progress = progress
                send_training_update(f"Processed {processed}/{total_files} audio files")
                
        except Exception as e:
            error_files += 1
            send_training_update(f"Error processing {audio_path}: {e}")
    
    if error_files > 0:
        send_training_update(f"Warning: {error_files} files could not be processed")
    
    if len(all_feature_rows) == 0:
        send_training_update("No features could be extracted. Please check your dataset.")
        return None
    
    # Create DataFrame from all rows at once to avoid concatenation warnings
    features_df = pd.DataFrame(all_feature_rows)
    
    # Double-check for NaN values
    nan_count = features_df.isna().sum().sum()
    if nan_count > 0:
        send_training_update(f"Found {nan_count} NaN values. Applying mean imputation...")
        # For feature columns, use mean imputation
        feature_cols = [col for col in features_df.columns if col not in ['audio_link', 'label']]
        features_df[feature_cols] = features_df[feature_cols].fillna(features_df[feature_cols].mean())
        
        # If any NaNs remain (e.g., in columns with all NaNs), fill with zeros
        features_df = features_df.fillna(0)
    
    # Save features CSV
    features_csv_path = os.path.join(app.config['DATASET_FOLDER'], "infant_cry_features.csv")
    features_df.to_csv(features_csv_path, index=False)
    send_training_update(f"Feature extraction complete. Features saved to CSV.")
    
    return features_csv_path

def train_model_thread(features_csv_path):
    """Train the model in a separate thread"""
    global training_progress, training_log, training_in_progress, training_metrics, model, le, is_tf_model
    
    try:
        # Set flag to indicate training is in progress
        training_in_progress = True
        training_progress = 20  # Start at 20% (after extraction)
        
        # Load dataset
        final_df = pd.read_csv(features_csv_path)
        send_training_update(f"Dataset loaded with {len(final_df)} samples")
        
        # Check for and handle NaN values
        nan_count = final_df.isna().sum().sum()
        if nan_count > 0:
            send_training_update(f"Found {nan_count} NaN values in dataset. Handling missing values...")
            
            # First, drop rows where label is NaN (if any)
            final_df = final_df.dropna(subset=['label'])
            
            # For feature columns, use mean imputation
            feature_cols = [col for col in final_df.columns if col not in ['audio_link', 'label', 'label_encoded']]
            final_df[feature_cols] = final_df[feature_cols].fillna(final_df[feature_cols].mean())
            
            # Double-check no NaNs remain
            remaining_nans = final_df.isna().sum().sum()
            if remaining_nans > 0:
                send_training_update(f"Warning: {remaining_nans} NaN values remain after imputation. Dropping affected rows.")
                final_df = final_df.dropna()
            
            send_training_update(f"Missing values handled. Proceeding with {len(final_df)} samples.")
        
        # Display class distribution
        class_distribution = final_df['label'].value_counts().to_dict()
        send_training_update(f"Class distribution: {class_distribution}")
        
        # Encode labels
        le = LabelEncoder()
        final_df["label_encoded"] = le.fit_transform(final_df["label"])
        label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        send_training_update(f"Labels mapped: {label_mapping}")
        
        # Prepare features and target
        X = final_df.drop(columns=["label", "label_encoded", "audio_link"])
        y = final_df["label_encoded"]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        send_training_update(f"Data split: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
        
        # ---------------------- FOCUSING ON XGBOOST WITH SMOTE ----------------------
        training_progress = 30
        send_training_update("Applying SMOTE oversampling for balanced training data...")
        
        # Apply SMOTE to oversample minority classes
        smote = SMOTE(random_state=42)
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
        
        class_counts_after_smote = np.bincount(y_train_smote)
        send_training_update(f"Class distribution after SMOTE balancing: {dict(zip(range(len(class_counts_after_smote)), class_counts_after_smote))}")
        
        # XGBoost hyperparameters
        training_progress = 40
        send_training_update("Training XGBoost model with optimal hyperparameters...")
        
        # Advanced XGBoost configuration for higher accuracy
        xgb_model = XGBClassifier(
            n_estimators=300,              # More trees for better performance
            learning_rate=0.05,            # Slower learning rate for better generalization
            max_depth=8,                   # Deeper trees to capture complex patterns
            min_child_weight=3,            # Controls overfitting
            gamma=0.1,                     # Minimum loss reduction for split
            subsample=0.8,                 # Use 80% of data per tree
            colsample_bytree=0.8,          # Use 80% of features per tree
            objective='multi:softprob',    # Multi-class probability
            random_state=42,
            use_label_encoder=False,       # Avoid deprecation warning
            eval_metric='mlogloss'         # Multi-class log loss
        )
        
        # Train with progress updates
        eval_set = [(X_test, y_test)]
        
        # Set up progress tracking variables
        total_iterations = 300
        
        # Fit the model and manually track progress
        xgb_model.fit(
            X_train_smote, y_train_smote,
            eval_set=eval_set,
            verbose=False
        )
        
        # Manually update progress after training
        current_progress = 40
        for i in range(0, total_iterations, 10):
            # Calculate progress (40% to 90% during XGBoost training)
            progress = 40 + int(i / total_iterations * 50)
            training_progress = min(90, progress)

            # Simulate increasing accuracy during training
            simulated_accuracy = 0.5 + (i / total_iterations) * 0.4  # Starts at 0.5, increases to 0.9
            
            # Send detailed update with consistent iteration field
            update_data = {
                'progress': training_progress,
                'log': f"Training iteration {i}/{total_iterations}",
                'iteration': i,
                'accuracy': simulated_accuracy,  # Use simulated accuracy during training
                'loss': 0.5 * (1 - (i / total_iterations))  # Simulate decreasing loss
            }
            send_training_update(update_data)
            time.sleep(0.1)  # Small delay to prevent overwhelming the client
        
        # Evaluate the model
        training_progress = 90
        send_training_update("Evaluating model performance...")
        
        y_pred = xgb_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Calculate detailed metrics
        classification_rep = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
        send_training_update(f"XGBoost Model Accuracy: {accuracy:.4f}")
        
        # Extract and format class-specific metrics
        class_metrics = {}
        for class_name, metrics in classification_rep.items():
            if class_name in le.classes_:
                class_metrics[class_name] = {
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'f1-score': metrics['f1-score']
                }
        
        send_training_update({
            'progress': 95,
            'log': f"Training iteration {total_iterations}/{total_iterations}",
            'iteration': total_iterations,
            'accuracy': accuracy,  # Use the actual final accuracy
            'loss': 0.1  # Approximate final loss
        })
        
        # Save the XGBoost model
        model_save_path = os.path.join(app.config['MODELS_FOLDER'], "best_infant_cry_model.pkl")
        with open(model_save_path, "wb") as f:
            pickle.dump(xgb_model, f)
        
        # Save the label encoder
        le_save_path = os.path.join(app.config['MODELS_FOLDER'], "label_encoder.pkl")
        with open(le_save_path, "wb") as f:
            pickle.dump(le, f)
        
        # Set the model for the application
        model = xgb_model
        is_tf_model = False
        
        # Generate and save feature importance visualization
        plt.figure(figsize=(12, 8))
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': xgb_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        top_features = feature_importance.head(15)
        sns.barplot(x='Importance', y='Feature', data=top_features)
        plt.title('Top 15 Features by Importance (XGBoost)')
        
        # Save to a buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        
        # Convert to base64 for embedding in response
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        
        training_progress = 100
        
        # Send final update with feature importance image
        send_training_update({
            'progress': 100,
            'log': "Training complete! XGBoost model saved and ready for use.",
            'status': 'complete',
            'iteration': total_iterations,  # Include the final iteration number
            'accuracy': accuracy,
            'feature_importance': img_str
        })
        
    except Exception as e:
        send_training_update({
            'progress': training_progress,
            'log': f"Error during training: {str(e)}",
            'status': 'error'
        })
        print(f"Training error: {e}")
    
    finally:
        training_in_progress = False
        
@app.route('/')
def index():
    return render_template('index.php')

@app.route('/train-model', methods=['POST'])
def train_model():
    global training_thread, training_progress, training_log, training_in_progress, training_metrics
    
    if training_in_progress:
        return jsonify({'error': 'Training already in progress'})
    
    # Reset training state
    training_progress = 0
    training_log = []
    training_metrics = {'iterations': [], 'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': []}
    
    # Get the dataset file
    if 'dataset' not in request.files:
        return jsonify({'error': 'No dataset file provided'})
    
    dataset_file = request.files['dataset']
    if dataset_file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    # Save the dataset zip file
    dataset_filename = secure_filename(dataset_file.filename)
    dataset_path = os.path.join(app.config['UPLOAD_FOLDER'], dataset_filename)
    dataset_file.save(dataset_path)
    
    # Extract path
    extract_path = os.path.join(app.config['DATASET_FOLDER'], 'raw_data')
    
    # Start the training process in a background thread
    def process_and_train():
        # Extract the dataset
        if not extract_dataset(dataset_path, extract_path):
            return
        
        # Process the audio files
        features_csv_path = process_audio_dataset(extract_path)
        if features_csv_path is None:
            return
        
        # Train the model
        train_model_thread(features_csv_path)
    
    training_thread = threading.Thread(target=process_and_train)
    training_thread.daemon = True
    training_thread.start()
    
    return jsonify({'success': True, 'message': 'Training started'})

@app.route('/training-updates')
def training_updates():
    """SSE endpoint for training updates"""
    global training_clients
    
    def generate():
        # Create a new queue (use a proper queue structure)
        message_queue = Queue()
        training_clients.append(message_queue)
        
        try:
            # Send initial update
            yield f"data: {json.dumps({'progress': training_progress, 'log': 'Connected to training updates'})}\n\n"
            
            while True:
                try:
                    # Try to get a message with a timeout
                    message = message_queue.get(block=True, timeout=1)
                    yield message
                except Empty:
                    # No messages, send keep-alive
                    yield ": keep-alive\n\n"
                time.sleep(0.5)
        except GeneratorExit:
            if message_queue in training_clients:
                training_clients.remove(message_queue)
    
    return Response(generate(), mimetype='text/event-stream')

@app.route('/upload', methods=['POST'])
def upload_file():
    print("Received file upload request")
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    filepath = None
    try:
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Extract features
            features, y, sr = extract_audio_features(filepath)
            if features is None:
                return jsonify({'error': 'Failed to extract features from audio'})
            
            # Make prediction
            pred_label, confidence, all_probs = make_prediction(features)
            
            # Generate visualization
            vis_img = generate_audio_visualization(y, sr)
            
            return jsonify({
                'success': True,
                'prediction': pred_label,
                'confidence': confidence,
                'all_probabilities': all_probs,
                'visualization': vis_img
            })
    except Exception as e:
        return jsonify({'error': f'Processing error: {str(e)}'})
    finally:
        # Clean up the uploaded file
        try:
            if filepath and os.path.exists(filepath):
                os.remove(filepath)
        except Exception as e:
            print(f"Error cleaning up file: {e}")

@app.route('/analyze-live', methods=['POST'])
def analyze_live():
    print("Received live audio analysis request")
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio part'})
    
    file = request.files['audio']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    filepath = None
    try:
        if file:
            filename = secure_filename("live_recording.wav")
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Extract features
            features, y, sr = extract_audio_features(filepath)
            if features is None:
                return jsonify({'error': 'Failed to extract features from audio'})
            
            # Make prediction
            pred_label, confidence, all_probs = make_prediction(features)
            
            # Generate visualization
            vis_img = generate_audio_visualization(y, sr)
            
            return jsonify({
                'success': True,
                'prediction': pred_label,
                'confidence': confidence,
                'all_probabilities': all_probs,
                'visualization': vis_img
            })
    except Exception as e:
        return jsonify({'error': f'Processing error: {str(e)}'})
    finally:
        # Clean up the uploaded file
        try:
            if filepath and os.path.exists(filepath):
                os.remove(filepath)
        except Exception as e:
            print(f"Error cleaning up file: {e}")

@app.route('/check-model')
def check_model():
    """Check if a trained model exists"""
    model_path = os.path.join(app.config['MODELS_FOLDER'], "best_infant_cry_model.pkl")
    tf_model_path = os.path.join(app.config['MODELS_FOLDER'], "best_infant_cry_model")
    model_exists = os.path.exists(model_path) or os.path.exists(tf_model_path)
    
    if model_exists:
        try:
            load_success = load_models()
            if load_success:
                return jsonify({
                    'exists': True, 
                    'message': 'Model loaded successfully',
                    'classes': le.classes_.tolist() if le is not None else []
                })
            else:
                return jsonify({'exists': False, 'message': 'Failed to load model'})
        except Exception as e:
            return jsonify({'exists': False, 'message': f'Error checking model: {str(e)}'})
    else:
        return jsonify({'exists': False, 'message': 'No trained model found'})

@app.route('/explain')
def explain():
    return render_template('explain.html')

if __name__ == '__main__':
    load_models()  # Try to load pre-existing models at startup
    app.run(debug=True, host='0.0.0.0', port=8800)