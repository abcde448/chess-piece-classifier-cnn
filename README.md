Chess Piece Image Classification using CNN
This project builds a Convolutional Neural Network (CNN) to classify chess pieces from images. It uses a labeled dataset of chess piece images and performs preprocessing, model training, evaluation, and visualization of performance metrics.

Dataset
Source: Custom dataset uploaded to Google Drive

Format: Images organized by folder per class (e.g., train/black_king/, train/white_queen/)

CSV: A chess_dataset_fixed.csv file maps image paths to labels.

Project Workflow
1.Data Preprocessing
Load and preprocess image data from folders.
Resize images to 64x64 and normalize pixel values.
Encode labels using LabelEncoder.
2.Model Building
Build a CNN using TensorFlow/Keras.
Use Conv2D, MaxPooling2D, Flatten, and Dense layers.
3.Model Training
Split data into training and test sets (80:20).
Train the model with validation.
4.Evaluation
Accuracy and loss graph plotted.
Confusion matrix and classification report generated.
5.Visualization
Display sample predictions with true and predicted labels.

Results
The model achieves strong classification accuracy across 13 different chess piece classes.
Model performance is visualized using confusion matrix and accuracy/loss plots.

Sample Visualization
plt.imshow(img)
plt.title(f"True: {true_label} | Pred: {predicted_label}")
Technologies Used
Python
Google Colab
TensorFlow / Keras
OpenCV
scikit-learn
seaborn, matplotlib, pandas, numpy

📬 Contact - 
Md. Rabbi Hasan |
Data Scientist
📧arafatrabbi668@gmail.com | LinkedIn https://www.linkedin.com/in/md-rabbi-hasan-3249652b1
Passionate about turning raw data into real-world insights through clear, compelling storytelling.
