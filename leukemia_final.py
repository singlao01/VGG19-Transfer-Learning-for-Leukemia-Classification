"""
================================================================================
LEUKEMIA DETECTION — COMPLETE SINGLE-CELL VERSION
CNN + K-FOLD CV + GRAD-CAM + SVM + RANDOM FOREST
Run this entire file as ONE cell in Jupyter, or run top to bottom.
================================================================================
"""

# ============================================================
# STEP 1: ALL IMPORTS FIRST
# ============================================================
import warnings
warnings.filterwarnings('ignore')

import os
import gc
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)
from sklearn.pipeline import Pipeline

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense,
    Dropout, BatchNormalization, Input
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs): return x

print("=" * 60)
print("All libraries imported successfully!")
print(f"TensorFlow : {tf.__version__}")
print(f"OpenCV     : {cv2.__version__}")
print(f"GPUs found : {len(tf.config.list_physical_devices('GPU'))}")
print("=" * 60)


# ============================================================
# STEP 2: CONFIGURATION  <-- ONLY CHANGE THIS PATH
# ============================================================

BASE_DATA_PATH = r'C:\Users\HP\OneDrive\Desktop\research dataset _ final'

IMG_HEIGHT    = 224
IMG_WIDTH     = 224
BATCH_SIZE    = 16
EPOCHS        = 30
LEARNING_RATE = 1e-4

# Set to 500 for a quick test run, None for full training
MAX_PER_CLASS = None

# Build fold paths AFTER os is imported
FOLDS = [
    {
        'name'    : 'fold_0',
        'all_path': os.path.join(BASE_DATA_PATH, 'fold_0', 'fold_0', 'all'),
        'hem_path': os.path.join(BASE_DATA_PATH, 'fold_0', 'fold_0', 'hem'),
    },
    {
        'name'    : 'fold_1',
        'all_path': os.path.join(BASE_DATA_PATH, 'fold_1', 'fold_1', 'all'),
        'hem_path': os.path.join(BASE_DATA_PATH, 'fold_1', 'fold_1', 'hem'),
    },
    {
        'name'    : 'fold_2',
        'all_path': os.path.join(BASE_DATA_PATH, 'fold_2', 'fold_2', 'all'),
        'hem_path': os.path.join(BASE_DATA_PATH, 'fold_2', 'fold_2', 'hem'),
    },
]

print("\nChecking dataset paths...")
all_ok = True
for fold in FOLDS:
    for key in ['all_path', 'hem_path']:
        path = fold[key]
        if os.path.exists(path):
            n = len([f for f in os.listdir(path)
                     if f.lower().endswith(('.bmp', '.jpg', '.jpeg', '.png'))])
            print(f"  OK  {path}  ({n} images)")
        else:
            print(f"  MISSING --> {path}")
            all_ok = False

if not all_ok:
    raise FileNotFoundError(
        "\nOne or more folders not found. "
        "Please fix BASE_DATA_PATH above and re-run."
    )
print("All paths OK!\n")


# ============================================================
# STEP 3: IMAGE LOADING FUNCTION
# ============================================================

def load_images_from_folder(folder_path, label, img_size=(224, 224), max_images=None):
    """
    Loads images from a folder one by one (memory safe).
    label = 0 for ALL (cancerous), 1 for HEM (healthy)
    Returns float32 numpy arrays normalized to [0, 1].
    """
    images = []
    labels = []
    errors = 0
    valid_ext = ('.bmp', '.jpg', '.jpeg', '.png')

    all_files = os.listdir(folder_path)
    files = [f for f in all_files if f.lower().endswith(valid_ext)]

    if len(files) == 0:
        raise ValueError(f"No images found in: {folder_path}")

    if max_images is not None:
        files = files[:max_images]

    folder_name = os.path.basename(folder_path)

    for fname in tqdm(files, desc=f"Loading {folder_name}", leave=True):
        img_path = os.path.join(folder_path, fname)
        img = cv2.imread(img_path)

        if img is None:
            errors += 1
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # cv2.resize wants (width, height)
        img = cv2.resize(img, (img_size[1], img_size[0]))
        img = img.astype(np.float32) / 255.0

        images.append(img)
        labels.append(label)

    if errors > 0:
        print(f"  WARNING: Skipped {errors} unreadable files in {folder_name}")

    print(f"  Loaded {len(images)} images from {folder_name}")
    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int32)


def load_fold(fold_dict, img_size=(224, 224), max_per_class=None):
    """Loads both classes for a fold and returns shuffled X, y arrays."""
    print(f"\n--- Loading {fold_dict['name']} ---")

    X_all, y_all = load_images_from_folder(
        fold_dict['all_path'], label=0,
        img_size=img_size, max_images=max_per_class
    )
    X_hem, y_hem = load_images_from_folder(
        fold_dict['hem_path'], label=1,
        img_size=img_size, max_images=max_per_class
    )

    X = np.concatenate([X_all, X_hem], axis=0)
    y = np.concatenate([y_all, y_hem], axis=0)

    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    print(f"  Total: {len(X)} | ALL(cancer)={np.sum(y==0)} | HEM(healthy)={np.sum(y==1)}")
    print(f"  RAM usage: ~{X.nbytes / 1024**2:.0f} MB")
    return X, y


# ============================================================
# STEP 4: DATA AUGMENTATION
# ============================================================

def augment_images(X, y, augment_factor=1):
    """
    Generates augmented copies. augment_factor=1 doubles the dataset.
    """
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.85, 1.15],
        fill_mode='nearest'
    )

    aug_X = [X.copy()]
    aug_y = [y.copy()]

    for pass_num in range(augment_factor):
        batch = []
        for img in tqdm(X, desc=f"Augmenting pass {pass_num+1}", leave=False):
            augmented = next(datagen.flow(img[np.newaxis], batch_size=1))[0]
            batch.append(augmented)
        aug_X.append(np.array(batch, dtype=np.float32))
        aug_y.append(y.copy())

    return np.concatenate(aug_X, axis=0), np.concatenate(aug_y, axis=0)


# ============================================================
# STEP 5: CNN MODEL
# ============================================================

def build_cnn(input_shape=(224, 224, 3)):
    """4-block CNN. 'last_conv' and 'dense_features' layers are named
    for Grad-CAM and feature extraction respectively."""

    inputs = Input(shape=input_shape)

    # Block 1
    x = Conv2D(32, (3,3), activation='relu', padding='same', name='conv1_1')(inputs)
    x = BatchNormalization()(x)
    x = Conv2D(32, (3,3), activation='relu', padding='same', name='conv1_2')(x)
    x = MaxPooling2D(2, 2)(x)
    x = Dropout(0.25)(x)

    # Block 2
    x = Conv2D(64, (3,3), activation='relu', padding='same', name='conv2_1')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3,3), activation='relu', padding='same', name='conv2_2')(x)
    x = MaxPooling2D(2, 2)(x)
    x = Dropout(0.25)(x)

    # Block 3
    x = Conv2D(128, (3,3), activation='relu', padding='same', name='conv3_1')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3,3), activation='relu', padding='same', name='conv3_2')(x)
    x = MaxPooling2D(2, 2)(x)
    x = Dropout(0.25)(x)

    # Block 4 — last_conv used for Grad-CAM
    x = Conv2D(256, (3,3), activation='relu', padding='same', name='conv4_1')(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (3,3), activation='relu', padding='same', name='last_conv')(x)
    x = MaxPooling2D(2, 2)(x)

    # Classifier head
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu', name='dense_features')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1, activation='sigmoid', name='output')(x)

    model = Model(inputs, outputs, name='LeukemiaCNN')
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    return model

print("CNN Architecture:")
build_cnn().summary()


# ============================================================
# STEP 6: GRAD-CAM
# ============================================================

def make_gradcam_heatmap(img_array, model):
    """
    img_array: shape (1, H, W, 3), normalized.
    Returns heatmap array normalized to [0, 1].
    """
    grad_model = Model(
        inputs=model.inputs,
        outputs=[model.get_layer('last_conv').output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array)
        class_score = preds[:, 0]

    grads        = tape.gradient(class_score, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out     = conv_out[0]
    heatmap      = conv_out @ pooled_grads[..., tf.newaxis]
    heatmap      = tf.squeeze(heatmap)
    heatmap      = tf.maximum(heatmap, 0)
    heatmap      = heatmap / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_gradcam(img, heatmap, alpha=0.4):
    h_resized   = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    h_colored   = cv2.applyColorMap(np.uint8(255 * h_resized), cv2.COLORMAP_JET)
    h_colored   = cv2.cvtColor(h_colored, cv2.COLOR_BGR2RGB)
    img_uint8   = np.uint8(255 * img)
    return cv2.addWeighted(img_uint8, 1 - alpha, h_colored, alpha, 0)


def visualize_gradcam(model, X_test, y_test, n_samples=8,
                      save_path='gradcam_results.png'):
    n_each      = n_samples // 2
    cancer_idx  = np.where(y_test == 0)[0][:n_each]
    healthy_idx = np.where(y_test == 1)[0][:n_each]
    indices     = list(cancer_idx) + list(healthy_idx)

    fig, axes = plt.subplots(len(indices), 3, figsize=(12, len(indices) * 3))
    fig.suptitle("Grad-CAM — What the CNN Focuses On",
                 fontsize=14, fontweight='bold', y=1.01)

    col_titles = ['Original', 'Heatmap', 'Overlay']
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontweight='bold')

    for row, idx in enumerate(indices):
        img        = X_test[idx]
        true_label = 'ALL (Cancer)' if y_test[idx] == 0 else 'HEM (Healthy)'
        img_in     = img[np.newaxis]
        pred_prob  = float(model.predict(img_in, verbose=0)[0][0])
        pred_label = 'HEM (Healthy)' if pred_prob > 0.5 else 'ALL (Cancer)'
        correct    = true_label == pred_label

        heatmap     = make_gradcam_heatmap(img_in, model)
        superimposed= overlay_gradcam(img, heatmap)
        heatmap_vis = cv2.applyColorMap(
            np.uint8(255 * cv2.resize(heatmap, (IMG_WIDTH, IMG_HEIGHT))),
            cv2.COLORMAP_JET
        )
        heatmap_vis = cv2.cvtColor(heatmap_vis, cv2.COLOR_BGR2RGB)

        axes[row, 0].imshow(img)
        axes[row, 0].set_ylabel(f"True: {true_label}", fontsize=8)
        axes[row, 1].imshow(heatmap_vis)
        axes[row, 2].imshow(superimposed)
        axes[row, 2].set_xlabel(
            f"Pred: {pred_label} ({pred_prob*100:.1f}%)",
            fontsize=8, color='green' if correct else 'red'
        )
        for ax in axes[row]:
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Grad-CAM saved -> {save_path}")


# ============================================================
# STEP 7: HYBRID CLASSIFIERS (CNN features -> SVM / RF)
# ============================================================

def extract_features(model, X, batch_size=16):
    """Extracts 512-dim feature vectors from the dense_features layer."""
    feature_model = Model(
        inputs=model.input,
        outputs=model.get_layer('dense_features').output
    )
    parts = []
    for i in range(0, len(X), batch_size):
        parts.append(feature_model.predict(X[i:i+batch_size], verbose=0))
    return np.concatenate(parts, axis=0)


def run_hybrid_classifiers(X_tr_feat, y_train, X_te_feat, y_test):
    classifiers = {
        'SVM (RBF)': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(kernel='rbf', C=10, gamma='scale',
                        probability=True, random_state=42))
        ]),
        'Random Forest': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(
                n_estimators=200, random_state=42, n_jobs=-1))
        ]),
    }
    results = {}
    for name, clf in classifiers.items():
        print(f"  Training {name}...")
        clf.fit(X_tr_feat, y_train)
        y_pred = clf.predict(X_te_feat)
        y_prob = clf.predict_proba(X_te_feat)[:, 1]
        results[name] = {
            'accuracy' : accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall'   : recall_score(y_test, y_pred, zero_division=0),
            'f1'       : f1_score(y_test, y_pred, zero_division=0),
            'auc'      : roc_auc_score(y_test, y_prob),
            'cm'       : confusion_matrix(y_test, y_pred),
            'y_pred'   : y_pred,
            'y_prob'   : y_prob,
        }
        print(f"    Acc={results[name]['accuracy']*100:.2f}%  AUC={results[name]['auc']:.4f}")
    return results


# ============================================================
# STEP 8: PLOTTING HELPERS
# ============================================================

def plot_training_curves(history, fold_name, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Training Curves — {fold_name}', fontsize=13, fontweight='bold')

    axes[0].plot(history.history['accuracy'],     label='Train', color='#2ecc71', lw=2)
    axes[0].plot(history.history['val_accuracy'], label='Val',   color='#3498db', lw=2, ls='--')
    axes[0].set_title('Accuracy')
    axes[0].set_ylim([0, 1])
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history['loss'],     label='Train', color='#e74c3c', lw=2)
    axes[1].plot(history.history['val_loss'], label='Val',   color='#e67e22', lw=2, ls='--')
    axes[1].set_title('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_confusion_matrix(cm_arr, title, save_path=None):
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm_arr, annot=True, fmt='d', cmap='Blues',
                xticklabels=['ALL (Cancer)', 'HEM (Healthy)'],
                yticklabels=['ALL (Cancer)', 'HEM (Healthy)'],
                annot_kws={'size': 16, 'weight': 'bold'},
                linewidths=2, linecolor='black')
    plt.title(title, fontweight='bold', fontsize=13, pad=12)
    plt.ylabel('True Label', fontweight='bold')
    plt.xlabel('Predicted Label', fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_roc(roc_dict, title, save_path=None):
    """roc_dict = {model_name: (y_true, y_prob)}"""
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']
    plt.figure(figsize=(8, 6))
    for i, (name, (y_true, y_prob)) in enumerate(roc_dict.items()):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        plt.plot(fpr, tpr, lw=2, color=colors[i % len(colors)],
                 label=f'{name} (AUC={auc:.4f})')
    plt.plot([0,1],[0,1],'k--', lw=1, label='Random')
    plt.xlabel('False Positive Rate', fontweight='bold')
    plt.ylabel('True Positive Rate', fontweight='bold')
    plt.title(title, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================
# STEP 9: LOAD ALL FOLDS INTO MEMORY
# ============================================================
print("\n" + "="*60)
print("   LOADING ALL 3 FOLDS")
print("="*60)

fold_data = []
for fold in FOLDS:
    X, y = load_fold(fold,
                     img_size=(IMG_HEIGHT, IMG_WIDTH),
                     max_per_class=MAX_PER_CLASS)
    fold_data.append((X, y))

total = sum(len(y) for _, y in fold_data)
print(f"\nAll folds loaded. Total images in memory: {total}")


# ============================================================
# STEP 10: K-FOLD CROSS VALIDATION TRAINING LOOP
# ============================================================

all_results  = []
best_model   = None
best_acc     = 0.0
best_fold_i  = 0

callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=8,
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                      patience=4, min_lr=1e-7, verbose=1),
]

for test_i in range(3):
    train_is = [i for i in range(3) if i != test_i]

    print(f"\n{'='*60}")
    print(f"  FOLD {test_i} = TEST   |   FOLDS {train_is} = TRAIN")
    print(f"{'='*60}")

    X_train = np.concatenate([fold_data[i][0] for i in train_is], axis=0)
    y_train = np.concatenate([fold_data[i][1] for i in train_is], axis=0)
    X_test, y_test = fold_data[test_i]

    print(f"Train: {len(X_train)}  |  Test: {len(X_test)}")

    # Augment
    print("Applying augmentation (1 extra copy)...")
    X_train, y_train = augment_images(X_train, y_train, augment_factor=1)
    print(f"Augmented train size: {len(X_train)}")

    # Train CNN
    print("\nTraining CNN...")
    model = build_cnn((IMG_HEIGHT, IMG_WIDTH, 3))
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    plot_training_curves(history, f'Fold {test_i}',
                         save_path=f'curves_fold{test_i}.png')

    # CNN metrics
    y_prob_cnn = model.predict(X_test, verbose=0).flatten()
    y_pred_cnn = (y_prob_cnn > 0.5).astype(int)
    cnn_metrics = {
        'accuracy' : accuracy_score(y_test, y_pred_cnn),
        'precision': precision_score(y_test, y_pred_cnn, zero_division=0),
        'recall'   : recall_score(y_test, y_pred_cnn, zero_division=0),
        'f1'       : f1_score(y_test, y_pred_cnn, zero_division=0),
        'auc'      : roc_auc_score(y_test, y_prob_cnn),
        'cm'       : confusion_matrix(y_test, y_pred_cnn),
        'y_pred'   : y_pred_cnn,
        'y_prob'   : y_prob_cnn,
    }
    print(f"\nCNN -> Acc={cnn_metrics['accuracy']*100:.2f}%  "
          f"F1={cnn_metrics['f1']*100:.2f}%  "
          f"AUC={cnn_metrics['auc']:.4f}")

    if cnn_metrics['accuracy'] > best_acc:
        best_acc    = cnn_metrics['accuracy']
        best_model  = model
        best_fold_i = test_i

    # Feature extraction + hybrid classifiers
    print("\nExtracting features for hybrid classifiers...")
    X_tr_feat = extract_features(model, X_train)
    X_te_feat = extract_features(model, X_test)

    print("Training hybrid classifiers...")
    hybrid = run_hybrid_classifiers(X_tr_feat, y_train, X_te_feat, y_test)

    # ROC curves
    plot_roc(
        {
            'CNN'           : (y_test, y_prob_cnn),
            'SVM (RBF)'     : (y_test, hybrid['SVM (RBF)']['y_prob']),
            'Random Forest' : (y_test, hybrid['Random Forest']['y_prob']),
        },
        title=f'ROC Curves — Fold {test_i}',
        save_path=f'roc_fold{test_i}.png'
    )

    fold_result = {'CNN': cnn_metrics}
    fold_result.update(hybrid)
    all_results.append(fold_result)

    # Free RAM before next fold
    del X_train, y_train, X_tr_feat, X_te_feat
    gc.collect()
    print(f"\nFold {test_i} complete.")


# ============================================================
# STEP 11: RESULTS TABLE
# ============================================================
model_names = ['CNN', 'SVM (RBF)', 'Random Forest']
metrics     = ['accuracy', 'precision', 'recall', 'f1', 'auc']

print("\n" + "="*75)
print("   FINAL RESULTS — 3-FOLD CROSS VALIDATION")
print("="*75)
print(f"{'Model':<20} {'Metric':<12} {'F0':>8} {'F1':>8} {'F2':>8} {'Mean':>8} {'Std':>8}")
print("-"*75)

summary = {}
for mname in model_names:
    summary[mname] = {}
    for metric in metrics:
        vals = [all_results[f][mname][metric] for f in range(3)]
        mean, std = np.mean(vals), np.std(vals)
        summary[mname][metric] = {'mean': mean, 'std': std}
        print(f"{mname:<20} {metric:<12} "
              f"{vals[0]:>8.4f} {vals[1]:>8.4f} {vals[2]:>8.4f} "
              f"{mean:>8.4f} {std:>8.4f}")
    print()
print("="*75)


# ============================================================
# STEP 12: AGGREGATE CONFUSION MATRIX
# ============================================================
agg_cm = sum(all_results[f]['CNN']['cm'] for f in range(3))
plot_confusion_matrix(
    agg_cm,
    title='Aggregate Confusion Matrix — CNN (3-Fold CV)',
    save_path='aggregate_cm.png'
)

tn, fp, fn, tp = agg_cm.ravel()
print(f"Sensitivity (Recall): {tp/(tp+fn)*100:.2f}%")
print(f"Specificity:          {tn/(tn+fp)*100:.2f}%")


# ============================================================
# STEP 13: GRAD-CAM ON BEST MODEL
# ============================================================
print(f"\nGenerating Grad-CAM using best CNN (fold {best_fold_i})...")
X_best, y_best = fold_data[best_fold_i]
visualize_gradcam(best_model, X_best, y_best,
                  n_samples=8, save_path='gradcam_results.png')


# ============================================================
# STEP 14: SAVE MODEL AND RESULTS
# ============================================================
best_model.save('leukemia_cnn_best.h5')
print("Best model saved -> leukemia_cnn_best.h5")

save_data = {
    mname: {
        metric: {
            'mean' : round(summary[mname][metric]['mean'], 6),
            'std'  : round(summary[mname][metric]['std'],  6)
        }
        for metric in metrics
    }
    for mname in model_names
}
with open('results_summary.json', 'w') as f:
    json.dump(save_data, f, indent=2)
print("Results saved -> results_summary.json")

print("\n" + "="*60)
print("  ALL DONE. Output files generated:")
print("="*60)
print("  curves_fold{0,1,2}.png    training curves per fold")
print("  roc_fold{0,1,2}.png       ROC curves per fold")
print("  aggregate_cm.png          aggregate confusion matrix")
print("  gradcam_results.png       Grad-CAM heatmaps")
print("  leukemia_cnn_best.h5      saved best CNN model")
print("  results_summary.json      all metrics as JSON")
print("="*60)
