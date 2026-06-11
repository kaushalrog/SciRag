# SciRAG-UQ 🔬

**Uncertainty-Aware Retrieval-Augmented Generation for Multi-Source Scientific Literature Synthesis**

> *A trustworthy scientific QA framework that dynamically adjusts source trustworthiness and refuses to answer when supporting evidence is insufficient.*

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![Conference](https://img.shields.io/badge/Target-BDA_2026-red)
![Status](https://img.shields.io/badge/Status-Active_Research-success)

---

## 📖 Overview

Scientific literature synthesis requires extreme factual precision. Despite the success of Retrieval-Augmented Generation (RAG), current systems struggle when retrieved evidence is contradictory, outdated, or fundamentally insufficient to answer a query. In such cases, LLMs often hallucinate plausible but incorrect answers.

**SciRAG-UQ** addresses this by shifting the paradigm from *“always generating an answer”* to *“knowing when to remain silent.”* 

This repository contains the codebase and experimental framework to prove that combining evidence fusion, confidence estimation, calibration, and abstention significantly reduces hallucinations in scientific QA while maintaining answer quality.

## ✨ Core Research Contributions

1. **Multi-Source Evidence Fusion:** Dynamic weighting of retrieved chunks across arXiv, Semantic Scholar, and local corpora based on historical source reliability, retrieval scores, and recency.
2. **Hybrid Confidence Estimation:** A formalized uncertainty signal combining:
   - *Retrieval Confidence (RC)*: Embedding similarity scores.
   - *Agreement Confidence (AC)*: Cross-source semantic agreement.
   - *Consistency Confidence (CC)*: Mean cosine similarity of multiple generated answers.
3. **Confidence Calibration:** Application of Platt Scaling and Isotonic Regression to map raw confidence scores to true probabilities.
4. **Calibrated Abstention Mechanism:** A $\tau$ thresholding logic that triggers a refusal to answer when the evidence is deemed insufficient, minimizing hallucination risk.

## 🚀 Repository Structure

The architecture is explicitly designed as an ML research framework prioritizing experimental evidence over software engineering boilerplate.

```text
scirag-uq/
├── data/
│   ├── raw/                 # Downloaded PDFs categorized by domain
│   ├── chroma_db/           # Persistent vector embeddings
│   ├── benchmark.json       # 3-tier gold-standard QA dataset
│   └── calibration_set.json # Confidence labels for Platt scaling
├── src/
│   ├── ingestion/           # arXiv scraping and PDF chunking pipeline
│   ├── retrieval/           # Base retrievers and formal fusion equations
│   ├── embeddings/          # all-MiniLM-L6-v2 embedders and chunkers
│   ├── uncertainty/         # Estimation, Calibration, and Abstention modules
│   ├── generation/          # SciRAG-UQ and Baseline LLM generators
│   └── evaluation/          # Metrics (ECE, AUROC) and Significance testing
├── scripts/                 # Reproducibility scripts
├── notebooks/               # Reliability diagrams and result analysis
└── paper/                   # BDA 2026 LaTeX manuscript
```

## 🛠️ Getting Started

### 1. Installation

Clone the repository and install the heavy ML dependencies (PyTorch, SentenceTransformers, ChromaDB, etc.):

```bash
git clone git@github.com:kaushalrog/scirag-uq.git
cd scirag-uq
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Data Collection & Ingestion (Phase 1)

Download the target corpus (~100-150 papers) from arXiv and extract metadata:
```bash
python3 src/ingestion/arxiv_collector.py
```

Chunk the downloaded PDFs and upsert the embeddings into ChromaDB:
```bash
python3 src/ingestion/pipeline.py
```

### 3. Running the Experiments

Once the `benchmark.json` is populated with gold evidence, execute the full evaluation suite across all baselines (Standard RAG, RAG + Confidence, SciRAG-UQ) and ablations:

```bash
./scripts/run_all_experiments.sh
```

## 📊 Evaluation Metrics

Our framework evaluates performance beyond standard RAG metrics, prioritizing uncertainty quantification:
- **Expected Calibration Error (ECE)**
- **Area Under the ROC Curve (AUROC)**
- **Abstention Precision & Recall**
- **Hallucination Rate**
- **Faithfulness**

## 📝 License & Citation

MIT License. If you use this framework or dataset in your research, please cite the upcoming BDA 2026 paper.
