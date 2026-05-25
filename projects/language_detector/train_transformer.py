import os
import sys
import json
import torch
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    DataCollatorWithPadding
)
from huggingface_hub import HfApi
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro')
    acc = accuracy_score(labels, predictions)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def main():
    corpus_path = "/Users/olumideola/Desktop/olaverse-ai/projects/language_detector/lid_corpus.json"
    model_name = "castorini/afriberta_large"
    output_dir = "/Users/olumideola/Desktop/olaverse-ai/projects/language_detector/lid_neural_5_model"
    
    if not os.path.exists(corpus_path):
        print(f"Error: Corpus file not found at {corpus_path}")
        sys.exit(1)
        
    print("Loading training corpus...")
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)
        
    classes = sorted(list(corpus.keys())) # ['eng', 'hau', 'ibo', 'pcm', 'yor']
    class_to_id = {cls: idx for idx, cls in enumerate(classes)}
    print(f"Classes: {classes}")
    print(f"Class-to-ID mapping: {class_to_id}")
    
    texts = []
    labels = []
    for lang, sentences in corpus.items():
        for sent in sentences:
            texts.append(sent)
            labels.append(class_to_id[lang])
            
    # Split into train and validation (90% train, 10% val)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.1, random_state=42, stratify=labels
    )
    
    print(f"Train size: {len(train_texts)}, Validation size: {len(val_texts)}")
    
    # Load tokenizer
    print(f"Loading tokenizer: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Dataset preparation
    train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
    val_dataset = Dataset.from_dict({"text": val_texts, "label": val_labels})
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=128)
        
    print("Tokenizing datasets...")
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_val = val_dataset.map(tokenize_function, batched=True)
    
    # Determine device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Training device selected: {device.upper()}")
    
    # Load model
    print(f"Loading model: {model_name}...")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(classes))
    model.to(device)
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=3e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=100,
        push_to_hub=False,
        report_to="none"
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics
    )
    
    print("\nStarting training...")
    trainer.train()
    
    print("\nEvaluation results:")
    eval_results = trainer.evaluate()
    print(eval_results)
    
    # Save the model and config locally
    print(f"\nSaving model to {output_dir}...")
    trainer.save_model(output_dir)
    
    # Save class mapping alongside config
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "r") as f:
        config_data = json.load(f)
    config_data["id2label"] = {str(i): cls for idx, (cls, i) in enumerate(class_to_id.items())}
    config_data["label2id"] = class_to_id
    with open(config_path, "w") as f:
        json.dump(config_data, f, indent=2)
        
    print("✅ Model successfully trained and saved!")
    
    # Optional upload to Hugging Face Hub
    upload = input("\nDo you want to upload the trained model to the Hugging Face Hub (olaverse/lid-neural-5)? (y/n): ").strip().lower()
    if upload == 'y':
        token = os.environ.get("HF_TOKEN")
        if not token:
            print("Error: HF_TOKEN environment variable not set. Please set it before uploading.")
            sys.exit(1)
            
        api = HfApi()
        repo_id = "olaverse/lid-neural-5"
        print(f"Creating/verifying repo '{repo_id}' on Hugging Face Hub...")
        try:
            api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, token=token)
        except Exception as e:
            print(f"Warning: {e}")
            
        print("Uploading model files...")
        try:
            api.upload_folder(
                folder_path=output_dir,
                repo_id=repo_id,
                repo_type="model",
                token=token
            )
            print("✅ Successfully uploaded to Hugging Face Hub!")
        except Exception as e:
            print(f"❌ Upload failed: {e}")

if __name__ == "__main__":
    main()
