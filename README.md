Step 1: Problem Definition
Goal: Classify microscopic blood cell images into:
Cancerous (Leukemia)
Benign (Normal)
Key challenge: Class imbalance (5.5:1) and high clinical risk of false negatives
Priority metric: Sensitivity (recall for cancer cases)

Step 2: Dataset Preparation
Total images: 3,256
Classes:
Early, Pre, Pro → merged into Cancerous
Benign → Non-cancerous
Two dataset versions:
Original images
Segmented images (background removed)

Step 3: Data Preprocessing
Resize images (typically 224×224 for VGG19)
Normalize pixel values
Apply data augmentation (if used)
Handle imbalance using:
Class weights
Cancerous: 0.59
Benign: 3.23

Step 4: Model Selection (Transfer Learning)
Base model: VGG19 (pretrained on ImageNet)
Freeze convolutional layers (feature extractor)
Add custom classification head:
Global Average Pooling
Batch Normalization
Dense layers (512 → 256 → 128)
Dropout (0.5, 0.3, 0.2)
Final Sigmoid layer (binary output)

Step 5: Model Compilation
Loss function: Binary Cross-Entropy (with class weights)
Optimizer: Adam (learning rate = 0.001)
Metrics tracked:
Accuracy
Sensitivity
Specificity
F1-score
AUC-ROC

Step 6: Training Strategy
Use 3-Fold Stratified Cross-Validation
Ensures class balance in each fold
Train on:
2 folds → Training
1 fold → Validation
Repeat 3 times and average results

Step 7: Threshold Optimization
Instead of default 0.5:
Use Youden’s J Statistic
Helps maximize:
Sensitivity + Specificity

Step 8: Model Evaluation

Evaluate using:

Sensitivity (Recall) → MOST IMPORTANT
Accuracy
AUC-ROC
F1-score
Specificity

Final Results:
Sensitivity: 98.69% (Original)
Accuracy: 97.39%
AUC: 0.994
Strong performance especially in detecting cancer cases

Step 9: Comparison Study

Compare:
Original images vs Segmented images
Observation:
Original performed slightly better in sensitivity
Segmented improved specificity

Step 10: Additional Techniques (if implemented)
Grad-CAM → Model interpretability
Hybrid models:
CNN + SVM
CNN + Random Forest

Step 11: Conclusion
Transfer learning with VGG19 is effective
High sensitivity ensures minimal missed cancer cases
Suitable for clinical decision support systems
