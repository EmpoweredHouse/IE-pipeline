# IE Pipeline: Dialogue Act Classification for Content Filtering

## Project Overview

**Goal:** Create an efficient dialogue act classification system to filter meaningful content from conversational noise in meeting transcripts and dialogues.

**Core Task:** Binary classification (Content vs Non-Content) using dialogue act labels as intermediate representation.

## Dataset & Labels

### Primary Dataset: MRDA Corpus
- **Source:** `wylupek/mrda-corpus` on HuggingFace
- **Size:** 108,202 utterances across train/validation/test splits
- **Domain:** Meeting conversations (75 meetings, 52 unique speakers)
- **Language:** English
- **Label Granularity:** 3 levels (Basic: 5 labels, General: 12 labels, Full: 52 labels)

### Label Strategy: General DA (12 labels)
Based on analysis in `MRDA_LABEL_ANALYSIS.md`:
- **Content Labels (71.8%):** s, qy, qw, qh, qrr, qr, qo (statements, questions, task-related)
- **Non-Content Labels (28.2%):** b, fh, %, fg, h (backchannels, floor management, disruptions)
- **Optimal Ratio:** Best content/non-content distinction for filtering applications

## Technical Approach

### Model Architecture
**Selected Model:** DistilBERT-base-uncased
- **Size:** 66M parameters (~250MB)
- **Performance:** 60% faster inference than BERT-base
- **Efficiency:** Optimal for production deployment and batch processing

### Fine-tuning Strategy: Parameter-Efficient Fine-Tuning (PEFT)
**Method:** LoRA (Low-Rank Adaptation)
- **Adapter Size:** ~6MB per task-specific adapter
- **Training Speed:** ~15 minutes per epoch on modern GPU
- **Memory:** 5GB GPU memory during training, 3GB during inference

### Training Configuration
```yaml
model:
  name: "distilbert-base-uncased"
  max_length: 256
  
peft:
  method: "lora"
  r: 16
  lora_alpha: 32
  target_modules: ["q_lin", "v_lin", "k_lin", "out_lin"]
  lora_dropout: 0.1
  
training:
  batch_size: 32
  learning_rate: 2e-4
  num_epochs: 3
  warmup_ratio: 0.1
```

## Data Processing Pipeline

### Input Format
- **Text:** Raw utterance text from meeting transcripts
- **Context:** Individual utterances (no conversation context initially)
- **Labels:** General DA labels mapped to binary content/non-content

### Label Mapping Strategy
```python
CONTENT_LABELS = {
    's': 'content',      # Statement
    'qy': 'content',     # Yes-No-question  
    'qw': 'content',     # Wh-Question
    'qh': 'content',     # Rhetorical Question
    'qrr': 'content',    # Or-Clause
    'qr': 'content',     # Or Question
    'qo': 'content'      # Open-ended Question
}

NON_CONTENT_LABELS = {
    'b': 'non-content',  # Continuer
    'fh': 'non-content', # Floor Holder
    '%': 'non-content',  # Interrupted/Abandoned
    'fg': 'non-content', # Floor Grabber
    'h': 'non-content'   # Hold Before Answer
}
```

### Data Splits
- **Training:** 75.1k samples (69.4%)
- **Validation:** 16.4k samples (15.2%)
- **Test:** 16.7k samples (15.4%)

## Training Strategy

### Phase 1: Multi-class Dialogue Act Classification
1. **Objective:** Train on 12-class General DA classification
2. **Purpose:** Learn nuanced dialogue act representations
3. **Output:** DistilBERT + LoRA adapter for 12 DA classes

### Phase 2: Binary Content Filtering
1. **Objective:** Map 12 DA classes to binary content/non-content
2. **Purpose:** Create production-ready content filter
3. **Options:** 
   - Direct mapping (no additional training)
   - Additional binary classification layer

## Evaluation Metrics

### Multi-class Performance (General DA)
- **Primary:** Macro F1-score across 12 dialogue act classes
- **Secondary:** Per-class precision/recall for imbalanced classes
- **Target:** >90% macro F1 on MRDA test set

### Binary Classification Performance (Content vs Non-Content)
- **Primary:** Binary accuracy and F1-score
- **Secondary:** Precision/recall for content detection
- **Target:** >94% accuracy, >90% F1 for both classes

### Efficiency Metrics
- **Inference Speed:** <50ms per utterance on modern GPU
- **Throughput:** >400 utterances/second for batch processing
- **Memory:** <5GB GPU memory during inference

## Model Deployment Strategy

### Repository Structure
- **Code Repository:** Current IE-pipeline repo for training/evaluation code
- **Model Repository:** Separate HuggingFace model repository for trained adapters
- **Integration:** Seamless loading via `transformers` and `peft` libraries

### HuggingFace Hub Integration
```python
# Load trained model
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

base_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")
model = PeftModel.from_pretrained(base_model, "your-username/dialogue-act-classifier")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
```

## Technical Infrastructure

### Core Libraries
- **Training:** `transformers`, `peft`, `accelerate`
- **Data:** `datasets`, `pandas`, `scikit-learn`
- **Evaluation:** `evaluate`, `seaborn`, `matplotlib`
- **Deployment:** `fastapi`, `uvicorn` (future production API)

### Development Environment
- **Platform:** Jupyter Notebooks for experimentation
- **GPU Requirements:** 6GB+ VRAM (RTX 3070/4060 Ti or better)
- **Memory:** 16GB+ system RAM recommended
- **Storage:** 10GB for models, data, and checkpoints

## Expected Outcomes

### Performance Targets
- **Dialogue Act Classification:** 90-92% macro F1 on MRDA test set
- **Content Filtering:** 94-96% binary accuracy
- **Speed:** Real-time processing of meeting transcripts
- **Efficiency:** Minimal computational overhead for production use

### Use Cases
1. **Meeting Transcript Filtering:** Remove chitchat, backchannels, interruptions
2. **Content Summarization:** Focus summarization on meaningful dialogue acts
3. **Quality Assessment:** Measure content density in conversations
4. **Data Preprocessing:** Clean conversational data for downstream NLP tasks

## Future Extensions

### Multi-domain Adaptation
- **Additional Corpora:** Extend to customer service, educational dialogues
- **Domain Adapters:** Train domain-specific LoRA adapters
- **Zero-shot Transfer:** Evaluate performance on unseen conversation types

### Conversation Context
- **Context Windows:** Incorporate previous utterances for better classification
- **Speaker Information:** Utilize speaker IDs for personalized models
- **Turn-taking Patterns:** Model conversation flow dynamics

---

**Status:** Ready for implementation phase with DistilBERT + LoRA on MRDA corpus for General DA classification.

**Next Steps:** Implement training pipeline in Jupyter notebook with HuggingFace ecosystem.
