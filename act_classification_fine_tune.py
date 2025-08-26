import json
import os
from datetime import datetime

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    TaskType
)

import warnings
warnings.filterwarnings('ignore')



def detect_device():
    """Detect best available device with fallback strategy"""
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps" 
    else:
        device = "cpu"
    
    return device


def preprocess_function(examples, tokenizer, label2id):
    """Tokenize text and encode labels"""
    tokens = tokenizer(
        examples["text"],
        truncation=True,
        padding=False,  # Will pad later in DataCollator
        max_length=128,  # Actual max length is 96
        return_tensors=None
    )
    tokens["labels"] = [label2id[label] for label in examples["general_da"]]
    return tokens


def calculate_class_weights(train_labels, label2id, max_weight, min_weight):
    label_counts = pd.Series(train_labels).value_counts().sort_index()
    id_counts = dict([(label2id[idx], count) for idx, count in label_counts.items()])

    class_weights = [1 / x for x in id_counts.values()]
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

    # Scale weights to MAX_WEIGHT and MIN_WEIGHT
    curr_max_weight = class_weights_tensor.max()
    curr_min_weight = class_weights_tensor.min()
    class_weights_tensor = (class_weights_tensor - curr_min_weight) / (curr_max_weight - curr_min_weight) * (max_weight - min_weight) + min_weight

    return class_weights_tensor


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class AdvancedTrainer(Trainer):
    def __init__(self, loss_type="focal_weighted", class_weights=None, focal_gamma=2.0, loss_reduction="mean", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_type = loss_type
        self.class_weights = class_weights
        
        if loss_type == "focal_weighted":
            self.loss_fn = FocalLoss(alpha=class_weights, gamma=focal_gamma, reduction=loss_reduction)
        elif loss_type == "focal":
            self.loss_fn = FocalLoss(gamma=focal_gamma, reduction=loss_reduction)
        elif loss_type == "ce_weighted":
            self.loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
        elif loss_type == "ce":
            self.loss_fn = torch.nn.CrossEntropyLoss()
        else:
            self.loss_fn = torch.nn.CrossEntropyLoss()
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        loss = self.loss_fn(logits.view(-1, logits.shape[-1]), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    # NOTE Accuracy is the same as micro F1
    macro_f1 = f1_score(labels, predictions, average='macro')
    micro_f1 = f1_score(labels, predictions, average='micro')
    weighted_f1 = f1_score(labels, predictions, average='weighted')
    accuracy = accuracy_score(labels, predictions)
    
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "weighted_f1": weighted_f1
    }


def setup_experiment_logging(experiment_name, hyperparams):
    """Setup timestamped logging directory for experiments"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results_ac{experiment_name}_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


def comprehensive_evaluation(trainer, tokenized_datasets, id2label, results_dir, hyperparams):
    # ===== WRITE HYPERPARAMETERS =====
    with open(f"{results_dir}/hyperparams.json", 'w') as f:
        json.dump(hyperparams, f, indent=2)

    # ===== GET PREDICTIONS =====
    predictions = trainer.predict(tokenized_datasets["validation"])
    y_true = predictions.label_ids
    y_pred = np.argmax(predictions.predictions, axis=1)
    
    # ===== OVERALL METRICS =====
    metrics = {
        'accuracy': round(accuracy_score(y_true, y_pred), 4),
        'macro_f1': round(f1_score(y_true, y_pred, average='macro'), 4),
        'weighted_f1': round(f1_score(y_true, y_pred, average='weighted'), 4),
    }
    
    # ===== PER-CLASS METRICS =====
    class_report = classification_report(
        y_true, y_pred,
        target_names=[id2label[i] for i in range(len(id2label))],
        output_dict=True
    )
    
    def round_nested_dict(d, decimals=4):
        """Recursively round all numeric values in nested dictionary"""
        if isinstance(d, dict):
            return {k: round_nested_dict(v, decimals) for k, v in d.items()}
        elif isinstance(d, (int, float)):
            return round(d, decimals) if isinstance(d, float) else d
        else:
            return d
    
    class_report = round_nested_dict(class_report)
    
    # ===== CONFUSION MATRIX =====
    cm = confusion_matrix(y_true, y_pred)
    
    # ===== TRAINING PROGRESS EXTRACTION =====
    training_progress = []
    if hasattr(trainer, 'state') and hasattr(trainer.state, 'log_history'):
        # Separate evaluation and training logs
        eval_logs = [log for log in trainer.state.log_history if 'eval_loss' in log]
        train_logs = [log for log in trainer.state.log_history if 'loss' in log and 'eval_loss' not in log]
        
        # Map training steps to training loss
        train_loss_map = {}
        for log in train_logs:
            step = log.get('step', 0)
            if step > 0:
                train_loss_map[step] = log.get('loss', 0)
        
        # Match evaluation checkpoints with training loss
        for log in eval_logs:
            eval_step = log.get('step', 0)
            
            # Find closest training loss (≤ eval_step)
            training_loss = 0
            if train_loss_map:
                valid_train_steps = [s for s in train_loss_map.keys() if s <= eval_step]
                if valid_train_steps:
                    closest_train_step = max(valid_train_steps)
                    training_loss = train_loss_map[closest_train_step]
            
            progress_entry = {
                'step': eval_step,
                'training_loss': round(training_loss, 4),
                'validation_loss': round(log.get('eval_loss', 0), 4),
                'accuracy': round(log.get('eval_accuracy', 0), 4),
                'macro_f1': round(log.get('eval_macro_f1', 0), 4),
                'micro_f1': round(log.get('eval_micro_f1', 0), 4),
                'weighted_f1': round(log.get('eval_weighted_f1', 0), 4)
            }
            training_progress.append(progress_entry)
    
    # ===== DETAILED 12-CLASS ANALYSIS =====
    detailed_analysis = {}
    for class_id in range(len(id2label)):
        class_name = id2label[class_id]
        
        # Calculate TP/FP/FN/TN for this class
        tp = cm[class_id, class_id]
        fp = cm[:, class_id].sum() - tp
        fn = cm[class_id, :].sum() - tp
        tn = cm.sum() - tp - fp - fn
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Calculate distribution percentages
        true_count = (y_true == class_id).sum()
        pred_count = (y_pred == class_id).sum()
        true_pct = (true_count / len(y_true)) * 100
        pred_pct = (pred_count / len(y_pred)) * 100
        
        # Store detailed analysis
        detailed_analysis[class_name] = {
            'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
            'precision': round(precision, 4), 'recall': round(recall, 4), 'f1': round(f1, 4),
            'true_pct': round(true_pct, 4), 'pred_pct': round(pred_pct, 4), 
            'diff_pct': round(pred_pct - true_pct, 4)
        }
        
    # ===== CLASS DISTRIBUTION ANALYSIS =====
    true_dist = pd.Series(y_true).value_counts().sort_index()
    pred_dist = pd.Series(y_pred).value_counts().sort_index()
    
    # ===== SAVE COMPREHENSIVE RESULTS =====
    results = {
        'overall_metrics': metrics,
        'per_class_metrics': class_report,
        'detailed_class_analysis': detailed_analysis,
        'training_progress': training_progress,
        'confusion_matrix': cm.tolist(),
        'true_distribution': true_dist.to_dict(),
        'pred_distribution': pred_dist.to_dict()
    }
    
    with open(f"{results_dir}/evaluation.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


def fine_tune_model(params):
    device = detect_device()

    dataset = load_dataset("wylupek/mrda-corpus")
    train_labels = [sample['general_da'] for sample in dataset['train']]
    unique_labels = list(set(train_labels))
    unique_labels.sort()
    
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: label for label, idx in label2id.items()}

    model = AutoModelForSequenceClassification.from_pretrained(
        params["model_name"],
        num_labels=params["num_labels"],
        problem_type="single_label_classification"
    )

    tokenizer = AutoTokenizer.from_pretrained(params["model_name"])
    tokenized_datasets = dataset.map(
        lambda x: preprocess_function(x, tokenizer, label2id),
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing"
    )
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        target_modules=params["target_modules"],
        lora_dropout=params["lora_dropout"],
        r=params["lora_r"],
        lora_alpha=params["lora_alpha"],
    )
    advanced_peft_model = get_peft_model(model, lora_config)

    class_weights_tensor = calculate_class_weights(train_labels, label2id, params["max_weight"], params["min_weight"]).to(device)

    total_steps = params["no_epochs"] * np.ceil(len(tokenized_datasets["train"]) / params["batch_size"])
    eval_steps = total_steps // params["no_epochs"] // 6
    save_steps = eval_steps * 3
    logging_steps = total_steps // params["no_epochs"] // 12
    advanced_training_args = TrainingArguments(
        output_dir="./advanced_checkpoints",
        save_strategy="steps",
        eval_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",  # Optimize for macro F1, perfect for imbalanced data
        greater_is_better=True,
        report_to=None,
        remove_unused_columns=False, # Required for custom trainer
        dataloader_num_workers=0,  # Important for MPS compatibility
        eval_steps=eval_steps,
        save_steps=save_steps,
        logging_steps=logging_steps,
        learning_rate=params["learning_rate"],
        num_train_epochs=params["no_epochs"],
        per_device_train_batch_size=params["batch_size"],
        per_device_eval_batch_size=params["batch_size"],
        warmup_steps=int(params["warmup_steps"] * total_steps), # Prevents overfitting, should be ~15% of total steps for imbalanced data
        weight_decay=params["weight_decay"],
    )
    advanced_trainer = AdvancedTrainer(
        loss_type=params["loss_type"],  # Focal loss for imbalanced data
        class_weights=class_weights_tensor, # Weights for imbalanced data
        model=advanced_peft_model,
        args=advanced_training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        focal_gamma=params["focal_gamma"],
        loss_reduction=params["loss_reduction"],
    )

    print(f"\nStarting advanced training with:")
    print(params)
    advanced_result = advanced_trainer.train()
    print(f"\nAdvanced training completed!")    
    print(f"Final train loss: {advanced_result.training_loss:.4f}")

    results_dir = setup_experiment_logging("no_1", params)
    comprehensive_evaluation(advanced_trainer, tokenized_datasets, id2label, results_dir)


if __name__ == "__main__":
    params = {
        "model_name": "distilbert-base-uncased",
        "num_labels": 12,

        "max_weight": 10,
        "min_weight": 0.3,

        "lora_dropout": 0.15,
        "lora_r": 64,
        "lora_alpha": 128,
        "target_modules": ["q_lin", "v_lin", "k_lin", "out_lin"],

        "no_epochs": 5,
        "batch_size": 32,
        "learning_rate": 0.00008,
        "warmup_steps": 0.15,
        "weight_decay": 0.05,

        "loss_type": "focal_weighted",
        "focal_gamma": 2.0,
        "loss_reduction": "mean",
    }
    fine_tune_model(params)
    print("Done")