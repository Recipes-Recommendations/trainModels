# Recipe Embeddings Model Training

This project fine-tunes a sentence transformer model to create embeddings for recipes based on their titles and ingredients. The model is trained to identify similar recipes using a contrastive learning approach.

## Overview

The training process involves:
1. Processing the recipe dataset to create pairs of similar and different recipes
2. Fine-tuning the `all-mpnet-base-v2` model using LoRA (Low-Rank Adaptation)
3. Training with cosine similarity loss
4. Pushing the fine-tuned model to Hugging Face Hub

## Prerequisites

- Python 3.12+
- CUDA-capable GPU
- Hugging Face account with write access
- AWS credentials configured for S3 access

## Dependencies

Install the required packages:
```bash
pip install accelerate bitsandbytes datasets loralib
pip install sentence-transformers transformers[torch] tqdm s3fs
```


## Usage

1. Set up environment variables:
```bash
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
```

2. Update S3 paths in the notebook:
```python
ORIGINAL_DATASET_S3_PATH = "s3://your-bucket/recipes_data.csv.zip"
RESULTS_S3_PATHS = [
    "s3://your-bucket/results-1.parquet",
    "s3://your-bucket/results-2.parquet"
]
```

3. Update model IDs:
```python
BASE_MODEL_ID = "sentence-transformers/all-mpnet-base-v2"
HF_MODEL_ID = "your-username/your-model-name"
```

4. Run the notebook cells in sequence

## Dataset Preparation

The training process uses two data sources:
1. Original recipe dataset (`recipes_data.csv.zip`) containing:
   - Recipe titles
   - Ingredients lists
   - Recipe links
   - Source information
   - Cooking directions

2. Processed dataset from the [createDataset Spark job](https://github.com/Recipes-Recommendations/createDataset) containing:
   - Similar recipe pairs
   - Different recipe pairs
   - Generated using TF-IDF and cosine similarity

The script combines these datasets to create:
- Positive pairs (similar recipes)
- Negative pairs (different recipes)
- Each pair includes:
  - Anchor recipe (title)
  - Positive/negative recipe (title + ingredients)
  - Label (1 for similar, 0 for different)

## Training Process

1. **Dataset Creation**
   - Creates Hugging Face dataset
   - Splits into train (49%), validation (21%), test (30%)
   - Filters for positive pairs when using cosine similarity loss
   - Renames columns for training

2. **Model Training**
   - Initializes base model
   - Applies LoRA configuration
   - Sets up training arguments
   - Creates trainer with loss function
   - Trains for specified steps
   - Evaluates on validation set
   - Pushes model to Hugging Face Hub

## Model Architecture

The model is based on `sentence-transformers/all-mpnet-base-v2` with the following modifications:

1. **LoRA (Low-Rank Adaptation)**
   - Freezes original model parameters
   - Adds low-rank matrices to specific layers
   - Only trains the new low-rank parameters
   - Reduces memory usage and training time

2. **Training Configuration**
   - Batch size: 48
   - Learning rate: 1e-6
   - Max steps: 200,000
   - FP16 precision
   - Cosine similarity loss

## Model Loading

After training, load the model using:
```python
from peft import PeftModel
from sentence_transformers import SentenceTransformer

model = PeftModel.from_pretrained(
    SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cuda"),
    "your-username/your-model-name"
)
```

## Output

The training process produces:
1. Fine-tuned model on Hugging Face Hub
2. Training logs with loss metrics
3. Validation results
4. Model checkpoints (if enabled)

## Notes

- The model uses LoRA to reduce memory requirements
- Training can be adjusted by modifying batch size and learning rate
- The model is trained to maximize similarity between similar recipes
- The final model can be used for recipe semantic search and recommendations
