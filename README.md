#leukemia-classification-vgg19#

1. Title & Overview (Markdown)
Leukemia Classification using VGG19 Transfer Learning
This notebook implements a deep learning model to classify blood cell images into cancerous and benign categories using transfer learning.

   Key Objective: Maximize sensitivity to reduce false negatives.

2. Imports
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

 3. Configuration 
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
BASE_DATA_PATH = "path/to/dataset"
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

4. Data Loading & Preprocessing
def load_data(data_path):
    """
    Loads and preprocesses dataset.

    Args:
        data_path (str): Dataset directory path

    Returns:
        train_data, val_data
    """
    
    if not os.path.exists(data_path):
        raise FileNotFoundError("Dataset path not found!")

    train_data = tf.keras.preprocessing.image_dataset_from_directory(
        data_path,
        validation_split=0.2,
        subset="training",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE
    )

    val_data = tf.keras.preprocessing.image_dataset_from_directory(
        data_path,
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE
    )

    return train_data, val_data
    
5. Model Building
def build_model():
    """
    Builds VGG19-based transfer learning model.
    """

    base_model = tf.keras.applications.VGG19(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3)
    )

    # Freeze pretrained layers
    base_model.trainable = False

    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    output = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=base_model.input, outputs=output)
    return model
   
6. Model Compilation & Training
def train_model(model, train_data, val_data):
    """
    Compiles and trains the model.
    """

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=EPOCHS
    )

    return history
   
7. Evaluation
def plot_history(history):
    """
    Plots training and validation accuracy.
    """
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.legend(['Train', 'Validation'])
    plt.title("Model Accuracy")
    plt.show()
   
8. Execution Cell (Final)
train_data, val_data = load_data(BASE_DATA_PATH)

model = build_model()
history = train_model(model, train_data, val_data)

plot_history(history)
Step 2: Clean Your Notebook (CRUCIAL)

Before submission:

Remove:
Debug prints
Random experiments
Duplicate cells
Unused imports

Keep:
Only final working pipeline

Step 3: Add Explanations Between Sections
After each major block, add a markdown explanation:

Example:
Why VGG19?
VGG19 is used due to its strong feature extraction capability and proven performance in medical imaging tasks.

Step 4: Add Outputs for Proof
Include:
Training graph
Confusion matrix
ROC curve
This shows execution proof (very important for grading)

Step 5: Execution Order Check
Click:
Kernel → Restart & Run All
-If it runs without error → perfect
-If not → fix dependencies/paths

Step 6: README Must Match Notebook

Your README should clearly say:
This is a Jupyter Notebook-based project

How to run:
jupyter notebook

Then open:
leukemia_final.ipynb
