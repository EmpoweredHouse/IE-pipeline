# Information Extraction Pipeline

> Modular pipeline for converting meeting transcripts into knowledge graph triples using lightweight, self-hosted models.

## Overview

This project implements a **7-step Information Extraction pipeline** that processes meeting transcripts, notes, and messages into knowledge-graph-ready format (**subject-predicate-object triples**). Designed as a cost-efficient, controllable alternative to direct "transcript → LLM" extraction approaches.

**Hypothesis:** Using lightweight, self-hosted models, we can achieve extraction quality on par with LLM-only baselines while improving privacy, cost, and latency.

### Key Features

- 🏗️ **Modular Architecture**: 7 independent, swappable pipeline steps
- 🔒 **Privacy-First**: All processing on-premise, no data egress required
- 💰 **Cost-Efficient**: Predictable costs, no token-metered pricing
- ⚡ **Low Latency**: Local inference, parallelizable processing
- 🎯 **Knowledge Graph Ready**: Direct output to structured triples
- 🔧 **Domain Adaptable**: Fine-tunable components for specific use cases

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/filipkozlowski/IE-pipeline.git
cd IE-pipeline

# Install dependencies with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

### Basic Usage

```python
# Coming soon - pipeline implementation in Jupyter notebook
# See: notebooks/smalltalk_detection_pipeline.ipynb
```

## Pipeline Architecture

| **Step** | **Purpose** | **Output** | **Status** |
|----------|-------------|------------|------------|
| **1. Parse & Clean** | Parse raw transcript into speaker-labeled utterances; normalize text | Clean, structured utterances | 🔄 Planned |
| **2. Content Filter** | Filter small talk/pleasantries while preserving work content | Content-focused text blocks | 🚧 **In Progress** |
| **3. Utterance Classification** | Classify into Action/Decision/Requirement/Info/Question | Labeled utterances | 🔄 Planned |
| **4. Named Entity Recognition** | Detect PER/ORG/PROJ/DATE/AMOUNT entities | Entity-tagged text | 🔄 Planned |
| **5. Relation Extraction** | Extract (subject, predicate, object) triples | Knowledge triples | 🔄 Planned |
| **6. Coreference Resolution** | Resolve pronouns to canonical entities | Resolved entity references | 🔄 Planned |
| **7. Aggregation & Output** | Format for knowledge graph ingestion | Final structured triples | 🔄 Planned |

### Current Focus: Step 2 - Content Filtering

- **Model**: `sentence-transformers/all-MiniLM-L6-v2` (22M parameters)
- **Method**: LoRA (Parameter Efficient Fine-Tuning) 
- **Context**: `[PREV] text [SEP] [CUR] text` format
- **Dataset**: AMI Meeting Corpus with functional annotations
- **Goal**: ≥95% precision small talk detection


## Development Status

🚧 **Phase 1: Step 2 - Content Filtering** - Implementing small talk detection with PEFT/LoRA

### Implementation Phases

**Phase 1: Content Filtering (Current)**
- [ ] Setup: Environment and model selection (MiniLM vs DistilBERT)
- [ ] Data: AMI corpus acquisition and preprocessing  
- [ ] Training: PEFT/LoRA small talk classifier
- [ ] Evaluation: Precision-optimized threshold selection
- [ ] Output: Deployable content filtering component

**Phase 2: Classification & NER**
- [ ] Step 3: Utterance classification (Action/Decision/Info/Question)
- [ ] Step 4: Named entity recognition (PER/ORG/PROJ/DATE)

**Phase 3: Knowledge Extraction**
- [ ] Step 5: Relation extraction and triple generation
- [ ] Step 6: Coreference resolution across utterances

**Phase 4: Integration & Evaluation**  
- [ ] Step 7: Output aggregation and formatting
- [ ] End-to-end pipeline integration
- [ ] Comparative evaluation vs LLM-only baseline

## Requirements

- **Python**: 3.11+ (optimized for performance and latest ML libraries)
- **Memory**: 8GB+ RAM recommended  
- **GPU**: Optional but recommended (6GB+ VRAM for training)
- **Storage**: ~5GB for models and data

## Research Goals

This project evaluates **three approaches** for meeting transcript → knowledge graph extraction:

- **(A) LLM-only**: Direct transcript → GPT-4/Claude → triples  
- **(B) Modular Pipeline**: 7-step lightweight model pipeline (this project)
- **(C) One-shot Extractor**: Single fine-tuned model for end-to-end extraction

**Success Metric**: Achieve extraction quality on par with LLM-only baseline while improving:
- **Privacy**: On-premise processing, no data egress
- **Cost**: Predictable inference costs vs token-metered pricing  
- **Latency**: Local processing vs external API calls

## Contributing

This is currently a research project focused on comparing IE approaches. Documentation and contribution guidelines will be added as components mature.

## License

MIT License - see LICENSE file for details.

---

*Part of a comparative study on cost-efficient, privacy-preserving information extraction from meeting transcripts.*
