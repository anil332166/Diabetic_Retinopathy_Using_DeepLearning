!pip install kaggle



!kaggle datasets download -d tanlikesmath/diabetic-retinopathy-resized



!kaggle datasets download -d tanlikesmath/diabetic-retinopathy-resized -p /path/to/download/directory



import kagglehub
# Download latest version
path = kagglehub.dataset_download("tanlikesmath/diabetic-retinopathy-resized")
print("Path to dataset files:", path)



import os
files=os.listdir(path)
files



for i in range(4):
  file_path = os.path.join(path, files[i])  # Access the first file in the list
  print("Path to a specific file:", file_path)



# linux command to unzip the .zip folder and then remove the .zip folder
! unzip '*.zip' && rm -f '*.zip'



! cd ./root/.cache/kagglehub/datasets/tanlikesmath/diabetic-retinopathy-resized/versions/7/resized_train



! ls



# Importing required moules and libraries. Numpy for scientific calculation, pandas for cleaning dataset
# Pyplot from matplotlib and seaborn for plotting various graph
# CV2 is used for performing image manipulation task
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import shuffle
import cv2
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split



df_train = pd.read_csv('/content/drive/MyDrive/Datasets/trainLabels.csv')
df_test = pd.read_csv('/content/drive/MyDrive/Datasets/trainLabels_cropped.csv')
SEED = 42
df_test.head()



x = df_train['image']
y = df_train['level']
x, y = shuffle(x, y, random_state=SEED)
print(df_train.head())



train_x, valid_x, train_y, valid_y = train_test_split(x, y, test_size=0.15,stratify=y, random_state=SEED)
print(train_x.shape, train_y.shape, valid_x.shape, valid_y.shape)
train_y.hist()
valid_y.hist()



import cv2
IMG_SIZE = 64
fig = plt.figure(figsize=(25, 16))
# display 5 images from each class
for class_id in train_y.unique():
    j=0
    for i, (idx, row) in enumerate(df_train.loc[df_train['level'] == class_id].sample(5, random_state=SEED).iterrows()):
        path=f"/content/drive/MyDrive/Datasets/resized_train/resized_train/{row['image']}.jpeg"
        image = cv2.imread(path)
        if os.path.exists(path):
          print(path)
          ax = fig.add_subplot(5, 5, class_id * 5 +j+ 1, xticks=[], yticks=[],)
          image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
          image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
          plt.imshow(image)
          ax.set_title('Label: %d-%d-%s' % (class_id, idx, row['image']) )
          j+=1



fig = plt.figure(figsize=(25, 16))
for class_id in sorted(train_y.unique()):
    for i, (idx, row) in enumerate(df_train.loc[df_train['level'] == class_id].sample(5, random_state=SEED).iterrows()):
        ax = fig.add_subplot(5, 5, class_id * 5 + i + 1, xticks=[], yticks=[])
        path=f"/content/drive/MyDrive/Datasets/resized_train/resized_train/{row['image']}.jpeg"
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

        plt.imshow(image, cmap='gray')
        ax.set_title('Label: %d-%d-%s' % (class_id, idx, row['image']) )



dpi = 80 #inch
path=f"/content/drive/MyDrive/Datasets/resized_train/resized_train/28036_left.jpeg"
image = cv2.imread(path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
height, width = image.shape
print(height, width)
SCALE=1
figsize = (width / float(dpi))/SCALE, (height / float(dpi))/SCALE
fig = plt.figure(figsize=figsize)
plt.imshow(image, cmap='gray')



fig = plt.figure(figsize=(25, 16))
for class_id in sorted(train_y.unique()):
    for i, (idx, row) in enumerate(df_train.loc[df_train['level'] == class_id].sample(5, random_state=SEED).iterrows()):
        ax = fig.add_subplot(5, 5, class_id * 5 + i + 1, xticks=[], yticks=[])
        path=f"/content/drive/MyDrive/Datasets/resized_train/resized_train/{row['image']}.jpeg"
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        image=cv2.addWeighted ( image,4, cv2.GaussianBlur( image , (0,0) , IMG_SIZE/10) ,-4 ,128) # the trick is to add this line,
        # adding this above line uses, gaussian blur and apply the filter on the original image  which makes the boundary clearly visible, thus enhancing the edges of the grayscale image
        plt.imshow(image, cmap='gray')
        ax.set_title('Label: %d-%d-%s' % (class_id, idx, row['image']) )



def load_ben_color(path, sigmaX=10):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image=cv2.addWeighted ( image,4, cv2.GaussianBlur( image , (0,0) , sigmaX) ,-4 ,128)
return image



import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
# Assuming you already have train_images and train.csv
# Load your data
train_df = pd.read_csv('/content/drive/MyDrive/Datasets/trainLabels.csv')
# Define constants
IMG_SIZE = 224
SEED = 42
# Apply filter to images
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), IMG_SIZE/10), -4, 128)
    return np.expand_dims(image, axis=-1)  # Add channel dimension
# Preprocess images and labels
X = np.array([preprocess_image(f"/content/drive/MyDrive/Datasets/resized_train/resized_train/{image_id}.jpeg") for image_id in train_df['image']])
y = train_df['level']
# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
# Split data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED)



import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
# Assuming you already have train_images and train.csv
# Load your data
train_df = pd.read_csv('/content/drive/MyDrive/Datasets/trainLabels.csv')
# Define constants
IMG_SIZE = 224
SEED = 42
# Apply filter to images
def preprocess_image(image_path):
    # Add a check for image loading success
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image: {image_path}")  # Print the problematic path
        return None  # Or handle the error in another way, e.g., return a placeholder image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), IMG_SIZE/10), -4, 128)
    return np.expand_dims(image, axis=-1)  # Add channel dimension
# Preprocess images and labels
X = [preprocess_image(f"/content/drive/MyDrive/Datasets/resized_train/resized_train/{image_id}.jpeg") for image_id in train_df['image']]
# Filter out None values (failed image loads)
X = [img for img in X if img is not None]
X = np.array(X)
y = train_df['level']
# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
# Split data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED)



!pip install imbalanced-learn



import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from tqdm import tqdm
# Assuming you already have train_images and train.csv
# Load your data
train_df = pd.read_csv('/content/trainLabels.csv')
# Define constants
IMG_SIZE = 224
SEED = 42
# Apply filter to images
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), IMG_SIZE/10), -4, 128)
    return np.expand_dims(image, axis=-1)  # Add channel dimension
# Preprocess images and labels
X = []
for image_id in tqdm(train_df['image'], desc="Preprocessing images"):
    image = preprocess_image(f"/content/resized_train/resized_train/{image_id}.jpeg")
    X.append(image)
X = np.array(X)
y = train_df['level']
# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
# Split data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED)
# Perform SMOTE on training data
smote = SMOTE(random_state=SEED)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train.reshape(-1, IMG_SIZE * IMG_SIZE), y_train)
# Reshape the resampled X_train data back to the original shape
X_train_resampled = X_train_resampled.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
# Now X_train_resampled and y_train_resampled contain the resampled training data
# X_val and y_val remain the same as the validation data



len(X_train_resampled)



num_samples = X.shape[0]
height = X.shape[1]
width = X.shape[2]
channels = X.shape[3]
X_reshaped = X.reshape(num_samples, -1)  # Flatten the height, width, and channels dimensions
# Save the reshaped array as a CSV file
np.savetxt('images.csv', X_reshaped, delimiter=',')



import tensorflow.keras.layers as layers
import tensorflow.keras.models as models
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import GridSearchCV
# Define a function to create your Keras model
def create_model(optimizer='adam'):
    IMG_SIZE = 224  # Adjust according to your input image size
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),  # Dropout layer to reduce overfitting
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),  # Dropout layer to reduce overfitting
        layers.Dense(5, activation='softmax')  # 5 classes for diabetic retinopathy severity levels
    ])
    # Compile the model with the specified optimizer
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model
# Define custom wrapper class
class KerasClassifierWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, build_fn=create_model, **kwargs):
        self.build_fn = build_fn
        self.kwargs = kwargs
        self.model = None
    def fit(self, X, y, **fit_params):
        self.model = self.build_fn(**self.kwargs)
        self.model.fit(X, y, **fit_params)
        return self
    def predict(self, X):
        return self.model.predict(X)
    def score(self, X, y, **kwargs):
        _, accuracy = self.model.evaluate(X, y, **kwargs)
        return accuracy
# Create an instance of the custom wrapper
model = KerasClassifierWrapper(build_fn=create_model, epochs=10, batch_size=32, verbose=1)
# Define the grid search parameters
param_grid = {'optimizer': ['Adadelta', 'Adagrad', 'Adam', 'RMSprop', 'SGD']}
# Perform grid search
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
grid_search_result = grid_search.fit(X_train, y_train)
# Summarize results
print("Best: %f using %s" % (grid_search_result.best_score_, grid_search_result.best_params_))
means = grid_search_result.cv_results_['mean_test_score']
stds = grid_search_result.cv_results_['std_test_score']
params = grid_search_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))



    y_train_multi = np.empty(y_train.shape, dtype=y_train.dtype)
y_train_multi[4] = y_train[4]
for i in range(3, -1, -1):
    y_train_multi[i] = np.logical_or(y_train[i], y_train_multi[i+1])
print("Original y_train:", y_train.sum(axis=0))
print("Multilabel version:", y_train_multi.sum(axis=0))



import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
# Function to build the model
def build_model(optimizer='adam'):
    IMG_SIZE = 224  # Adjust according to your input image size
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),  # Dropout layer to reduce overfitting
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),  # Dropout layer to reduce overfitting
        layers.Dense(5, activation='softmax')  # 5 classes for diabetic retinopathy severity levels
    ])
    # Compile the model with the specified optimizer
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model
# Load your data
# Assuming you have X_train, X_val, y_train, y_val
# Define constants
SEED = 42
# Split data into train and validation sets
# Function to plot training history
def plot_history(history, optimizer):
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title(f'Model Performance with {optimizer} Optimizer')
    plt.xlabel('Epoch')
    plt.ylabel('Metric Value')
    plt.legend()
    plt.show()
# List of optimizers to try
optimizers = ['Adadelta', 'Adagrad', 'Adam', 'RMSprop', 'SGD']
# Train models with different optimizers and plot their training history
histories = {}
for opt in optimizers:
    print(f"Training model with {opt} optimizer...")
    model = build_model(opt)
    hist = model.fit(X_train_resampled, y_train_resampled, batch_size=32, epochs=10, verbose=1, validation_data=(X_val, y_val))
    histories[opt] = hist
    # Plot training history
    plot_history(hist, opt)



y_train_resampled.hist()



# Save the model
model.save("diabetic_retinopathy_model.keras")



from sklearn.metrics import cohen_kappa_score
def quadratic_weighted_kappa(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')
y_pred_probabilities = model.predict(X_val)
# Get the class with the highest probability for each sample
y_pred = np.argmax(y_pred_probabilities, axis=1)
kappa_score = quadratic_weighted_kappa(y_val, y_pred)
print("Quadratic weighted kappa:", kappa_score)



# RESNET
# Importing necessary modules and libraries
import tensorflow as tf
from tensorflow.keras import layers, models
# Define ResNet model
resnet_model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
     layers.Conv2D(512, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
     layers.Conv2D(1024, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
     layers.Conv2D(512, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
     layers.Conv2D(256, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(32, activation='relu'),
    layers.Dense(5, activation='softmax')
])
# Compile the ResNet model
resnet_model.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])
# Train the ResNet model
resnet_model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))
from sklearn.metrics import cohen_kappa_score
def quadratic_weighted_kappa(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')
y_pred_probabilities = model.predict(X_val)
# Get the class with the highest probability for each sample
y_pred = np.argmax(y_pred_probabilities, axis=1)
kappa_score = quadratic_weighted_kappa(y_val, y_pred)
print("Quadratic weighted kappa:", kappa_score)



# Define LeNet model
lenet_model = models.Sequential([
    layers.Conv2D(6, (5, 5), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(16, (5, 5), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(120, activation='relu'),
    layers.Dense(84, activation='relu'),
    layers.Dense(5, activation='softmax')
])
# Compile the LeNet model
lenet_model.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
# Train the LeNet model
lenet_model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))
from sklearn.metrics import cohen_kappa_score
def quadratic_weighted_kappa(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')
y_pred_probabilities = model.predict(X_val)
# Get the class with the highest probability for each sample
y_pred = np.argmax(y_pred_probabilities, axis=1)
kappa_score = quadratic_weighted_kappa(y_val, y_pred)
print("Quadratic weighted kappa:", kappa_score)



from tensorflow.keras.applications import DenseNet121
from tensorflow.keras import layers, models
densenet_model = DenseNet121(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights='imagenet')
for layer in densenet_model.layers:
    layer.trainable = False
flatten_layer = layers.Flatten()(densenet_model.output)
dense_layer_1 = layers.Dense(256, activation='relu')(flatten_layer)
dropout_layer = layers.Dropout(0.5)(dense_layer_1)
dense_layer_2 = layers.Dense(128, activation='relu')(dropout_layer)
output_layer = layers.Dense(5, activation='softmax')(dense_layer_2)
final_densenet_model = models.Model(inputs=densenet_model.input, outputs=output_layer)
final_densenet_model.compile(optimizer='adam',
                             loss='sparse_categorical_crossentropy',
                             metrics=['accuracy'])
final_densenet_model.fit(X_train_rgb, y_train, epochs=15, validation_data=(X_val_rgb, y_val))
from sklearn.metrics import cohen_kappa_score
def quadratic_weighted_kappa(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')
y_pred_probabilities = final_densenet_model.predict(X_val_rgb)
y_pred = np.argmax(y_pred_probabilities, axis=1)
kappa_score = quadratic_weighted_kappa(y_val, y_pred)
print("Quadratic weighted kappa:", kappa_score)



kappa_score



import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
# Step 2: Loading the Saved Model
model_path = '/content/diabetic_retinopathy_model.h5'
model = load_model(model_path)
# Step 3: Preparing Test Data
# Assuming you have test data stored in test_features.npy and test_labels.npy
test_features = np.load('test_features.npy')
test_labels = np.load('test_labels.npy')
# Step 4: Performing Inference
predictions = model.predict(test_features)
# Step 5: Evaluating Performance
# Assuming this is a classification task
accuracy = np.mean(np.argmax(predictions, axis=1) == np.argmax(test_labels, axis=1))
print("Accuracy:", accuracy)
















