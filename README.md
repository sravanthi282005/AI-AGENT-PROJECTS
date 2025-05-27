
# Healthcare and AI Projects

This repository contains three machine learning and deep learning projects:

1. **Healthcare Disease Prediction Using Machine Learning**
2. **IMDB Sentiment Analysis Using Deep Learning**
3. **Dog and Cat Image Classification Using CNN**

---

🐶🐱 Dog and Cat Image Classification Using CNN
A deep learning-based image classification project using Convolutional Neural Networks (CNNs) to accurately identify whether a given image contains a dog or a cat.

📌 1. Project Content
This project aims to:

Classify images into two categories: dogs or cats.

Utilize a Convolutional Neural Network (CNN) built with TensorFlow and Keras.

Load images from a directory and split them into training and validation sets.

Monitor training performance using metrics and plots.

Evaluate model performance on unseen data.

Serve as a base for transfer learning and real-world deployment scenarios.

This model can be integrated into:

Pet adoption apps.

Animal behavior monitoring systems.

Mobile-based pet recognition tools.

Smart surveillance and image tagging platforms.

💻 2. Project Code
python
Copy
Edit
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Constants
IMG_HEIGHT, IMG_WIDTH = 150, 150
BATCH_SIZE = 32

# Data Preparation
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    'path_to_dataset',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    'path_to_dataset',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

# CNN Model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compilation
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Training
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10
)

# Visualization
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title("Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid()
plt.show()
🧠 3. Key Technologies
Python: Core language for scripting and implementation.

TensorFlow / Keras: For building and training the CNN model.

ImageDataGenerator: For augmenting and loading image data.

Matplotlib: For visualizing training performance.

NumPy: For numerical operations and preprocessing.

📝 4. Description
This project demonstrates the use of a CNN model to identify whether an image contains a cat or a dog. CNNs are designed to automatically and adaptively learn spatial hierarchies of features through backpropagation by using multiple building blocks, such as convolution layers, pooling layers, and fully connected layers.

🧩 Dataset Assumptions
The dataset is expected to be structured like this:

markdown
Copy
Edit
path_to_dataset/
│
├── cats/
│   ├── cat1.jpg
│   ├── cat2.jpg
│   └── ...
│
└── dogs/
    ├── dog1.jpg
    ├── dog2.jpg
    └── ...
The model normalizes the images by rescaling pixel values from 0–255 to 0–1, resizes them to 150x150, and uses 80% for training and 20% for validation.

📈 5. Output
✅ Training Results
The model prints accuracy and loss values for both training and validation datasets.

Accuracy typically improves over epochs as the CNN learns features like fur patterns, ears, and face shape.

📊 Accuracy Plot
A line plot shows training vs. validation accuracy, helping identify overfitting or underfitting.

Example result after 10 epochs:

Train Accuracy: ~95%

Validation Accuracy: ~93%

📦 Final Results
The model is saved or exported using model.save("model_path.h5") for deployment.

You can predict using:

python
Copy
Edit
from tensorflow.keras.preprocessing import image
import numpy as np

img = image.load_img("sample.jpg", target_size=(150, 150))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)
prediction = model.predict(img_array)
print("Prediction: ", "Dog" if prediction[0][0] > 0.5 else "Cat")
🔬 6. Further Research
To enhance the model and expand its use, consider the following:

📍 Model Improvements
✅ Use transfer learning with models like ResNet50, InceptionV3, MobileNet.

✅ Apply data augmentation (rotation, zoom, flip) to improve generalization.

✅ Use dropout layers to prevent overfitting.

🔁 Pipeline Enhancements
Deploy the model using:

Flask or Streamlit for local web interface.

TensorFlow Lite or ONNX for mobile and edge devices.

🧪 Advanced Techniques
Convert to a multi-class classifier (e.g., dog breeds).

Add attention mechanisms to visualize which parts of the image the model focuses on.

Explore Grad-CAM for explainable AI.

Integrate real-time classification via webcam or smartphone camera.

📂 Summary
Component	Details
Model Type	Convolutional Neural Network
Dataset	Cats vs Dogs directory structure
Framework	TensorFlow / Keras
Input Shape	150x150 RGB Images
Output	Dog or Cat prediction
Accuracy	~93% (Validation)
Deployment Ready	Yes, can be exported as .h5


📚 7. References
TensorFlow and Keras Documentation
https://www.tensorflow.org/
https://keras.io/

Deep Learning with Python by François Chollet
A hands-on book that introduces CNNs and their applications using Keras.

Dogs vs. Cats Dataset – Kaggle
https://www.kaggle.com/c/dogs-vs-cats
Kaggle's popular image classification dataset used for binary classification tasks.

ImageDataGenerator API – TensorFlow Docs
https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator
For image loading, augmentation, and preprocessing.

CS231n: Convolutional Neural Networks for Visual Recognition – Stanford University
https://cs231n.github.io/convolutional-networks/
Provides foundational knowledge on CNN architecture and training.

Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow by Aurélien Géron
A comprehensive guide for implementing machine learning projects with deep learning frameworks.

Towards Data Science – Cat vs. Dog Classifier Using CNN
https://towardsdatascience.com/cats-vs-dogs-image-classification-with-deep-learning-4a3c8e518d87
An article providing an example of image classification with code snippets and explanation.

Matplotlib Documentation
https://matplotlib.org/stable/contents.html
Used for plotting training and validation accuracy/loss.

NumPy Documentation
https://numpy.org/doc/
Utilized for numerical and image array manipulation.

Grad-CAM: Visual Explanations from Deep Networks
https://arxiv.org/abs/1610.02391
For future research in visual explanations and model interpretability.


### 📌 Project Content
Develop a machine learning model to predict possible diseases based on patient health-related data.

### 🧠 Key Technologies
- Python
- Pandas, NumPy
- Scikit-learn
- Pickle

### 📝 Description
This project uses logistic regression to predict diseases based on features such as symptoms, age, and gender. The dataset is preprocessed, label encoded, and used to train the model.

### 💻 Code
```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import pickle

df = pd.read_csv("healthcare_dataset.csv")
df.drop(columns=['Name'], inplace=True)
label_encoders = {col: LabelEncoder().fit(df[col]) for col in df.select_dtypes(include=['object']).columns}
for col, le in label_encoders.items():
    df[col] = le.transform(df[col])

X = df.drop("Disease", axis=1)
y = df["Disease"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

with open('disease_predictor.pkl', 'wb') as f:
    pickle.dump(model, f)
```

### 📈 Output
- Accuracy score of the model.
- Classification report.
- A `.pkl` file to be deployed in healthcare systems.

### 🔬 Further Research
- Expand dataset
- Try ensemble models
- Build a web interface
- Integrate real-time data

---

## Project 2: IMDB Sentiment Analysis Using Deep Learning

### 📌 Project Content
Classify IMDB movie reviews as positive or negative using an LSTM-based deep learning model.

### 🧠 Key Technologies
- Python
- NLTK, BeautifulSoup
- Keras/TensorFlow
- Pandas

### 📝 Description
Movie reviews are cleaned, tokenized, and converted into padded sequences. An LSTM network learns to associate text with sentiment.

### 💻 Code
```python
import pandas as pd
import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

nltk.download('stopwords')
movie_reviews = pd.read_csv("/content/IMDB Dataset.csv")

def preprocess(text):
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text().lower()
    tokens = [word for word in text.split() if word not in stopwords.words('english')]
    return ' '.join(tokens)

movie_reviews['cleaned'] = movie_reviews['review'].apply(preprocess)

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(movie_reviews['cleaned'])
sequences = tokenizer.texts_to_sequences(movie_reviews['cleaned'])
X = pad_sequences(sequences, maxlen=200)
y = pd.get_dummies(movie_reviews['sentiment']).values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = Sequential([
    Embedding(5000, 64, input_length=200),
    LSTM(64),
    Dense(2, activation='softmax')
])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))
```

### 📈 Output
- Accuracy of the model.
- Predicts new reviews.
- Can be expanded to batch predictions.

### 🔬 Further Research
- Use pre-trained embeddings
- Try GRU or Bi-LSTM
- Deploy using APIs
- Add attention mechanisms

---

## Project 3: Dog and Cat Image Classification Using CNN

### 📌 Project Content
A CNN model to classify whether an image contains a dog or a cat.

### 🧠 Key Technologies
- Python
- TensorFlow / Keras
- NumPy
- Matplotlib

### 📝 Description
The dataset is split into training and validation sets with normalization and resizing. A CNN is trained using binary crossentropy.

### 💻 Code
```python
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

IMG_HEIGHT, IMG_WIDTH = 150, 150
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    'path_to_dataset',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    'path_to_dataset',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10
)

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.show()
```

### 📈 Output
- Accuracy and loss plots
- Image classification results
- Final model accuracy

### 🔬 Further Research
- Use ResNet/Inception
- Tune hyperparameters
- Use transfer learning
- Expand to more animal types
- Deploy using Flask or Streamlit
