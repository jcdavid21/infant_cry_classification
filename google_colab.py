import os
import pandas as pd

# Correct base directory path
base_dir = "/content/infant_cry_data/donateacry_corpus"
data = []

# Walk through directory structure and collect .wav files
for root, _, files in os.walk(base_dir):
    for file in files:
        if file.endswith(".wav"):
            file_path = os.path.join(root, file)
            label = os.path.basename(root)
            data.append([file_path, label])

# Create DataFrame from collected data
df = pd.DataFrame(data, columns=["audio_link", "label"])

# Fix the path for saving CSV in Colab
# In Colab, you should use /content/ as the base directory
csv_path = "/content/infant_cry_data.csv"  # Changed from /content/working/
df.to_csv(csv_path, index=False)

print(f"CSV file saved to: {csv_path}")

# Load and display the saved CSV file
dat = pd.read_csv(csv_path)  # Using the same path variable for consistency
print(f"Found {len(dat)} audio files")
dat.head(5)


def process_audio_csv(input_csv, output_csv, n_mfcc=13):
    try:
        # Load the CSV file with audio file paths
        df = pd.read_csv(input_csv)

        if "audio_link" not in df.columns:
            print("Error: 'audio_link' column not found in CSV!")
            return

        # Create column names for all the features
        feature_columns = [f"MFCC_{i+1}" for i in range(n_mfcc)] + \
                          ["Spectral_Centroid", "Zero_Crossing_Rate"] + \
                          [f"Chroma_{i+1}" for i in range(12)]

        # Initialize feature columns to None
        for col in feature_columns:
            df[col] = None

        # Process each audio file
        for i, row in df.iterrows():
            audio_path = row["audio_link"]
            try:
                # Load audio file
                y, sr = librosa.load(audio_path, sr=None)

                # Extract features
                mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc), axis=1)
                spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr), axis=1)
                zcr = np.mean(librosa.feature.zero_crossing_rate(y), axis=1)
                chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)

                # Combine all features
                features = np.concatenate([
                    mfccs, spectral_centroid.ravel(), zcr.ravel(), chroma
                ])

                # Save features to dataframe
                df.loc[i, feature_columns] = features

                # Progress update
                if i % 10 == 0:  # Print status every 10 files
                    print(f"Processed {i+1}/{len(df)} files")

            except Exception as e:
                print(f"Error processing {audio_path}: {e}")

        # Save the result
        df.to_csv(output_csv, index=False)
        print(f"Feature extraction complete. Saved to {output_csv}")

    except Exception as e:
        print(f"Error: {e}")

# Fix the path to match your Colab environment
process_audio_csv("/content/infant_cry_data.csv", "/content/infant_cry_features.csv")

# Import necessary libraries
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import tensorflow as tf
from tensorflow import keras
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

# Load the dataset
final_df = pd.read_csv("/content/infant_cry_features.csv")

# Display class distribution before balancing
print("Original class distribution:")
print(final_df['label'].value_counts())
print()

# Encode labels
le = LabelEncoder()
final_df["label_encoded"] = le.fit_transform(final_df["label"])
label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print(f"Labels mapped: {label_mapping}")

# Prepare features and target
X = final_df.drop(columns=["label", "label_encoded", "audio_link"])
y = final_df["label_encoded"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

# Check class distribution in training set
print("\nClass distribution in training set:")
print(pd.Series(y_train).value_counts())

# Plot class distribution
plt.figure(figsize=(10, 6))
sns.countplot(x=final_df['label'])
plt.title('Class Distribution in Original Dataset')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('/content/class_distribution.png')
plt.show()

# Visualization of confusion matrices function
def plot_confusion_matrix(y_true, y_pred, title, labels):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f'/content/{title.replace(" ", "_")}.png')
    plt.show()

# Function to evaluate and print results
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{model_name} Accuracy: {accuracy:.4f}")

    print(f"{model_name} Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    plot_confusion_matrix(y_test, y_pred, f"{model_name} Confusion Matrix", le.classes_)
    return accuracy, y_pred

# ---------------------- APPROACH 1: CLASS WEIGHTS ----------------------

print("\n==== APPROACH 1: USING CLASS WEIGHTS ====\n")

# Compute class weights inversely proportional to class frequencies
class_weights = dict(zip(
    np.unique(y_train),
    len(y_train) / (len(np.unique(y_train)) * np.bincount(y_train))
))
print("Class weights:", class_weights)

# Random Forest with class weights
rf_weighted = RandomForestClassifier(
    n_estimators=200,
    class_weight='balanced',
    random_state=42,
    max_depth=20
)
rf_weighted.fit(X_train, y_train)
rf_weighted_accuracy, rf_weighted_preds = evaluate_model(rf_weighted, X_test, y_test, "Random Forest (Weighted)")

# XGBoost with scale_pos_weight
# For multi-class, we need to apply weights differently than binary classification
xgb_weighted = XGBClassifier(
    random_state=42,
    scale_pos_weight=list(class_weights.values()),  # Approximate approach
    max_depth=6,
    learning_rate=0.1,
    n_estimators=200
)
xgb_weighted.fit(X_train, y_train)
xgb_weighted_accuracy, xgb_weighted_preds = evaluate_model(xgb_weighted, X_test, y_test, "XGBoost (Weighted)")

# ---------------------- APPROACH 2: SMOTE OVERSAMPLING ----------------------

print("\n==== APPROACH 2: USING SMOTE OVERSAMPLING ====\n")

# Apply SMOTE to oversample minority classes
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print("Class distribution after SMOTE:")
print(pd.Series(y_train_smote).value_counts())

# Train Random Forest on SMOTE-balanced data
rf_smote = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=20)
rf_smote.fit(X_train_smote, y_train_smote)
rf_smote_accuracy, rf_smote_preds = evaluate_model(rf_smote, X_test, y_test, "Random Forest (SMOTE)")

# Train XGBoost on SMOTE-balanced data
xgb_smote = XGBClassifier(random_state=42, max_depth=6, learning_rate=0.1, n_estimators=200)
xgb_smote.fit(X_train_smote, y_train_smote)
xgb_smote_accuracy, xgb_smote_preds = evaluate_model(xgb_smote, X_test, y_test, "XGBoost (SMOTE)")

# ---------------------- APPROACH 3: COMBINED SAMPLING ----------------------

print("\n==== APPROACH 3: COMBINED UNDER/OVERSAMPLING ====\n")

# Create a pipeline with over and undersampling
# First oversample minority classes to have more representation
# Then undersample majority class to reduce its dominance
sampling_pipeline = Pipeline([
    ('smote', SMOTE(sampling_strategy='auto', random_state=42)),
    ('undersampler', RandomUnderSampler(sampling_strategy='not minority', random_state=42))
])

X_train_combined, y_train_combined = sampling_pipeline.fit_resample(X_train, y_train)

print("Class distribution after combined sampling:")
print(pd.Series(y_train_combined).value_counts())

# Train Random Forest on combined-sampled data
rf_combined = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=20)
rf_combined.fit(X_train_combined, y_train_combined)
rf_combined_accuracy, rf_combined_preds = evaluate_model(rf_combined, X_test, y_test, "Random Forest (Combined Sampling)")

# Train XGBoost on combined-sampled data
xgb_combined = XGBClassifier(random_state=42, max_depth=6, learning_rate=0.1, n_estimators=200)
xgb_combined.fit(X_train_combined, y_train_combined)
xgb_combined_accuracy, xgb_combined_preds = evaluate_model(xgb_combined, X_test, y_test, "XGBoost (Combined Sampling)")

# ---------------------- APPROACH 4: NEURAL NETWORK WITH CLASS WEIGHTS ----------------------

print("\n==== APPROACH 4: NEURAL NETWORK WITH CLASS WEIGHTS ====\n")

# Convert class weights to the format expected by Keras
keras_class_weights = {int(cls): weight for cls, weight in class_weights.items()}

# Build the neural network
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.4),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(len(np.unique(y)), activation='softmax')
])

# Compile the model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Use early stopping
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# Reduce learning rate when plateauing
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=5,
    min_lr=0.0001
)

# Train the model with class weights
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping, reduce_lr],
    class_weight=keras_class_weights
)

# Evaluate the model
y_pred_nn = np.argmax(model.predict(X_test), axis=1)
nn_accuracy = accuracy_score(y_test, y_pred_nn)
print(f"Neural Network Accuracy: {nn_accuracy:.4f}")
print("Neural Network Classification Report:")
print(classification_report(y_test, y_pred_nn, target_names=le.classes_))
plot_confusion_matrix(y_test, y_pred_nn, "Neural Network Confusion Matrix", le.classes_)

# Plot training history
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.tight_layout()
plt.savefig('/content/training_history.png')
plt.show()

# ---------------------- FEATURE IMPORTANCE ANALYSIS ----------------------

print("\n==== FEATURE IMPORTANCE ANALYSIS ====\n")

# Check which features are most important for classification
feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Random Forest Importance': rf_weighted.feature_importances_,
    'XGBoost Importance': xgb_weighted.feature_importances_
})

feature_importances = feature_importances.sort_values('Random Forest Importance', ascending=False)
print(feature_importances.head(10))

# Plot feature importances
plt.figure(figsize=(12, 8))
sns.barplot(x='Random Forest Importance', y='Feature', data=feature_importances.head(15))
plt.title('Top 15 Features by Importance (Random Forest)')
plt.tight_layout()
plt.savefig('/content/feature_importance_rf.png')
plt.show()

plt.figure(figsize=(12, 8))
sns.barplot(x='XGBoost Importance', y='Feature', data=feature_importances.sort_values('XGBoost Importance', ascending=False).head(15))
plt.title('Top 15 Features by Importance (XGBoost)')
plt.tight_layout()
plt.savefig('/content/feature_importance_xgb.png')
plt.show()

# ---------------------- MODEL COMPARISON ----------------------

print("\n==== MODEL COMPARISON ====\n")

# Compare all models
models = {
    'Random Forest (Weighted)': rf_weighted_accuracy,
    'XGBoost (Weighted)': xgb_weighted_accuracy,
    'Random Forest (SMOTE)': rf_smote_accuracy,
    'XGBoost (SMOTE)': xgb_smote_accuracy,
    'Random Forest (Combined)': rf_combined_accuracy,
    'XGBoost (Combined)': xgb_combined_accuracy,
    'Neural Network': nn_accuracy
}

# Plot comparison
plt.figure(figsize=(12, 6))
sns.barplot(x=list(models.keys()), y=list(models.values()))
plt.title('Model Accuracy Comparison')
plt.xticks(rotation=45)
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig('/content/model_comparison.png')
plt.show()

# ---------------------- SAVE THE BEST MODEL ----------------------

# Determine the best model based on accuracy
best_model_name = max(models, key=models.get)
print(f"Best model: {best_model_name} with accuracy {models[best_model_name]:.4f}")

# Save the corresponding model
import pickle

if best_model_name == 'Random Forest (Weighted)':
    best_model = rf_weighted
elif best_model_name == 'XGBoost (Weighted)':
    best_model = xgb_weighted
elif best_model_name == 'Random Forest (SMOTE)':
    best_model = rf_smote
elif best_model_name == 'XGBoost (SMOTE)':
    best_model = xgb_smote
elif best_model_name == 'Random Forest (Combined)':
    best_model = rf_combined
elif best_model_name == 'XGBoost (Combined)':
    best_model = xgb_combined
elif best_model_name == 'Neural Network':
    best_model = model  # Already defined above

# Save the model and label encoder
if best_model_name != 'Neural Network':
    with open("/content/best_infant_cry_model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    print("Model saved as: /content/best_infant_cry_model.pkl")
else:
    best_model.save("/content/best_infant_cry_model")
    print("Model saved as: /content/best_infant_cry_model (TensorFlow format)")

with open("/content/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)
print("Label encoder saved as: /content/label_encoder.pkl")

# ---------------------- PREDICTION FUNCTION ----------------------

def predict_cry_type(audio_path, model_path="/content/best_infant_cry_model.pkl", le_path="/content/label_encoder.pkl"):
    # Function to extract features from a new audio file
    def extract_audio_features(audio_path, n_mfcc=13):
        import librosa
        import numpy as np
        try:
            y, sr = librosa.load(audio_path, sr=None)
            mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc), axis=1)
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr), axis=1)
            zcr = np.mean(librosa.feature.zero_crossing_rate(y), axis=1)
            chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
            features = np.concatenate([
                mfccs, spectral_centroid.ravel(), zcr.ravel(), chroma
            ])
            return features
        except Exception as e:
            print(f"Error extracting features from {audio_path}: {e}")
            return None

    # Load label encoder
    with open(le_path, "rb") as f:
        label_encoder = pickle.load(f)

    # Extract features
    features = extract_audio_features(audio_path)

    if features is None:
        return "Error extracting features"

    # Check if model is TensorFlow or sklearn
    if os.path.isdir(model_path):
        # TensorFlow model
        model = keras.models.load_model(model_path)
        prediction = np.argmax(model.predict([features.reshape(1, -1)])[0])
    else:
        # sklearn model
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        prediction = model.predict([features])[0]

    cry_type = label_encoder.inverse_transform([prediction])[0]
    return cry_type

# Example usage - update path to an actual file in your environment
example_file = "/content/infant_cry_data/donateacry_corpus/hungry/719bb382-a592-46b7-82d2-8b4a625263b7-1430376562788-1.7-m-48-hu.wav"

if os.path.exists(example_file):
    result = predict_cry_type(example_file,
                             model_path="/content/best_infant_cry_model.pkl" if best_model_name != 'Neural Network' else "/content/best_infant_cry_model",
                             le_path="/content/label_encoder.pkl")
    print(f"Predicted cry type: {result}")
else:
    print(f"Example file not found. Update the path to test prediction function.")