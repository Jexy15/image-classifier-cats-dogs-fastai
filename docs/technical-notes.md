## Technical Notes (Derived from TEORIA.pdf)

### Scope and context
These notes summarize core machine learning concepts used in a simple image classification workflow (example: cats vs dogs). The focus is on model training, transfer learning, and evaluation metrics that guide performance. Full reference: [docs/TEORIA.pdf](docs/TEORIA.pdf).

### Core concepts
**Model**
- A model maps an input (e.g., image) to an output label (e.g., cat or dog).
- The mapping depends on parameters (weights) that the training process learns from data.

**Weights and parameters**
- Weights control how input signals influence the output.
- During training, weights are adjusted to reduce prediction errors.

**Loss function**
- A loss function measures how far predictions are from the correct labels.
- The training goal is to minimize loss over the dataset.

**Error rate vs accuracy**
- Accuracy measures the fraction of correct predictions.
- Error rate is the fraction of incorrect predictions; it is often used as a metric because it aligns with the optimization goal (lower is better).

### Training workflow (high level)
1. Initialize a model with parameters (often pre-trained in transfer learning).
2. Run predictions on training data.
3. Compute loss or error rate.
4. Update weights to reduce error.
5. Repeat for multiple iterations (epochs) until performance stabilizes.

### Transfer learning
- Transfer learning starts from a model pre-trained on a large dataset.
- The model already captures generic visual patterns (edges, textures, shapes).
- You then fine-tune it on the specific task (cats vs dogs), often with fewer training examples and faster convergence.

### Overfitting vs underfitting
**Overfitting**
- The model performs well on training data but poorly on validation data.
- Indicates the model memorized training examples rather than learning general patterns.

**Underfitting**
- The model performs poorly on both training and validation data.
- Indicates the model is too simple or not trained long enough.

**Practical signal**
- Compare training and validation metrics: a big gap suggests overfitting.

### Gradient descent and SGD (Stochastic Gradient Descent)
- Gradient descent updates weights in the direction that reduces loss.
- SGD uses smaller batches, enabling faster iterations and better generalization in many cases.
- Each step adjusts weights slightly based on the current mini-batch error.

### Model capacity and generalization
- Bigger models can learn more complex patterns but risk overfitting.
- Validation metrics are essential to decide if the model generalizes.

### Evaluation and iteration
- Train, evaluate, and adjust hyperparameters (learning rate, regularization, data augmentation).
- Iterative refinement is key: evaluate errors, update the model, and repeat.

### Project metrics (from training runs)
- Model: ResNet34 (transfer learning)
- Dataset: Oxford-IIIT Pets (cats vs dogs labels)

| run | train_loss | valid_loss | error_rate |
| --- | --- | --- | --- |
| A | 0.179958 | 0.020274 | 0.006089 |
| B | 0.049542 | 0.012951 | 0.006089 |

### Practical takeaway for a portfolio project
- Show the full loop: data prep -> training -> validation -> evaluation -> iteration.
- Use simple, interpretable metrics (accuracy, error rate) and explain the tradeoffs.
- Highlight transfer learning as a practical technique to improve results with limited data.

### Suggested structure for a portfolio appendix
- Short technical summary (this document)
- Key metrics (accuracy, error rate, confusion matrix if available)
- Training decisions (transfer learning choice, SGD, validation strategy)

---

Note: These notes are based on OCR extraction of the PDF, which can introduce minor noise. If needed, I can refine this with a clean text version or manual corrections.
