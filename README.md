# Persian Topic Modeling

A modular framework for Persian topic modeling using BERTopic and ParsBERT.

## Overview

This project investigates topic modeling of Persian news documents using BERTopic and ParsBERT. It includes a baseline BERTopic model based on pretrained ParsBERT embeddings and a fine-tuned embedding model trained using supervised metric learning with BatchHard Triplet Loss.

## Features

- Persian text preprocessing
- ParsBERT-based document embeddings
- BERTopic topic modeling pipeline
- Supervised fine-tuning with Sentence Transformers
- BatchHard Triplet Loss
- Topic evaluation
- Topic statistics and analysis
- Interactive BERTopic visualizations
- Automatic topic labeling using Ollama (LLM)

## Methodology

### Baseline Model

The baseline pipeline uses pretrained ParsBERT embeddings followed by the standard BERTopic components:

**ParsBERT → UMAP → HDBSCAN → c-TF-IDF**

### Fine-tuned Model

For the improved model, ParsBERT is fine-tuned using supervised metric learning with BatchHard Triplet Loss. The resulting embedding model is then integrated into the same BERTopic pipeline:

**ParsBERT → BatchHard Triplet Loss → Fine-tuned Embeddings → UMAP → HDBSCAN → c-TF-IDF**

## Project Structure

```text
src/
├── preprocessing/
├── evaluation/
├── visualizer/
├── topic_labeling/
└── ...

models/
results/
figures/
notebooks/

requirements.txt
README.md
