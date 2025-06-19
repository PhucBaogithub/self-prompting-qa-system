# Self-Prompting QA System

**Enhanced Web Interface for Self-Prompting Large Language Models for Zero-Shot Open-Domain QA**

This project implements a comprehensive **Self-Prompting Question Answering System** with an intuitive web interface, featuring advanced clustering algorithms, topic coherence analysis, and model comparison capabilities.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Web Interface Guide](#web-interface-guide)
- [API Documentation](#api-documentation)
- [Technical Details](#technical-details)
- [Enhancements](#enhancements)
- [Citation](#citation)

## Features

### **Core Functionality**
- **QA Pair Generation**: Generate diverse question-answer pairs using Flan-T5 models
- **Advanced Clustering**: Multiple algorithms (K-Means, Hierarchical, Spectral, Gaussian Mixture)
- **Smart Example Selection**: Intelligent selection of representative examples from clusters  
- **Model Comparison**: Side-by-side comparison of different language models
- **Topic Coherence Analysis**: Advanced metrics for evaluating clustering quality

### **Web Interface**
- **Clean Modern UI**: Professional white/black minimalist design
- **Real-time Metrics**: Live display of clustering and inference metrics
- **Interactive Pipeline**: Step-by-step workflow visualization
- **Responsive Design**: Works seamlessly across different screen sizes
- **Question Diversity**: Enhanced question generation with 10+ categories

### **Advanced Metrics**
- **Clustering Metrics**: Silhouette Score, Topic Coherence, Calinski-Harabasz Index
- **Model Efficiency**: Speed analysis, quality comparison, efficiency scoring
- **Quality Assessment**: BLEU scores, semantic similarity, factual accuracy
- **Performance Insights**: Detailed model performance breakdowns

## Installation

### Prerequisites
- Python 3.8+
- 8GB+ RAM recommended
- Internet connection for model downloads

### 1. Clone Repository
```bash
git clone https://github.com/your-username/self-prompting-qa-system.git
cd self-prompting-qa-system
```

### 2. Create Virtual Environment
```bash
python -m venv self-prompting-env
source self-prompting-env/bin/activate  # On Windows: self-prompting-env\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Setup API Keys (Optional)
For enhanced functionality, create `./related_files/openai_api.txt` with your OpenAI API key.

## Quick Start

### Start Web Interface
```bash
python web_interface.py
```

Navigate to `http://localhost:5001` in your browser.

### Basic Workflow
1. **Generate**: Create QA pairs for topics like "science, technology, history"
2. **Cluster**: Group similar questions using advanced algorithms  
3. **Select**: Choose representative examples from each cluster
4. **Inference**: Test questions with multiple models and compare results

## Web Interface Guide

### 1. **Data Generation** 
- Enter topics (comma-separated): `science, technology, history, mathematics`
- Set QA pairs per topic: `3-5` recommended
- Click **Generate QA Pairs**
- View metrics: Generation time, topic diversity, question quality

### 2. **Clustering**   
- Configure cluster range: `[2, 3, 4, 5]` for optimal results
- Click **Perform Clustering**
- Monitor metrics: Silhouette Score (>0.4 excellent), Topic Coherence (>60% good)
- View cluster topics and sizes

### 3. **Example Selection** 
- Set max examples per cluster: `2-3` recommended  
- Click **Select Examples**
- Review selection metrics and cluster representatives

### 4. **Model Inference** 
- Enter test question or select from generated questions
- Choose model comparison mode: Flan-T5 vs DistilBERT
- Click **Run Inference**
- Analyze detailed comparison metrics

### 5. **Question Generation** 
- Generate diverse questions from clusters automatically
- 10+ question categories: Definition, Benefits, Challenges, Mechanics, Future, etc.
- Test generated questions with model comparison

## API Documentation

### Generate QA Pairs
```bash
POST /api/generate
{
  "topics": ["science", "technology"], 
  "num_pairs_per_topic": 3
}
```

### Perform Clustering  
```bash
POST /api/cluster
{
  "cluster_range": [2, 3, 4, 5]
}
```

### Select Examples
```bash
POST /api/select
{
  "max_examples_per_cluster": 2
}
```

### Run Inference
```bash
POST /api/inference
{
  "question": "What are the benefits of artificial intelligence?",
  "context": ""
}
```

## Technical Details

### Core Components
- **`self_prompting_pipeline.py`**: Main pipeline orchestration
- **`evaluation_metrics.py`**: Advanced metrics calculation  
- **`web_interface.py`**: Flask web server and UI
- **`api_utils.py`**: API utilities and helpers

### Clustering Algorithms
- **K-Means**: Fast, good for spherical clusters
- **Hierarchical**: Better for nested/tree-like structures
- **Spectral**: Excellent for non-convex clusters
- **Gaussian Mixture**: Handles overlapping clusters

### Models Supported
- **Flan-T5 (Small/Large)**: Google's instruction-tuned T5
- **DistilBERT**: Lightweight BERT variant
- **GPT-based models**: Via OpenAI API (optional)

## Recent Enhancements

### **Question Diversity Enhancement**
- **10 Question Categories**: Definition, Benefits, Challenges, Mechanics, Future, Comparison, Practical, Technical, Impact, Learning
- **Smart Category Selection**: AI-powered relevance matching
- **100% Unique Questions**: Zero repetition across categories
- **4 Templates per Category**: Maximum variety within each type

### **Topic Coherence Fix**
- **Enhanced Algorithm**: Robust topic inference with 150+ keywords
- **Fallback Mechanisms**: Multiple layers of topic detection
- **Improved Clustering**: Better topic preservation during data augmentation
- **Real-time Accuracy**: Values now correctly display 60-80% range

### **UI/UX Improvements**  
- **Minimalist Design**: Clean white background, black text
- **Professional Metrics**: Real-time clustering and inference statistics
- **Responsive Layout**: Optimized for all screen sizes
- **Enhanced Visualization**: Better charts and progress indicators

## Performance Benchmarks

### Typical Results
- **Generation**: 3-5 QA pairs/second
- **Clustering**: 2-10 seconds for 15-50 samples
- **Topic Coherence**: 60-85% (excellent >75%)  
- **Silhouette Score**: 0.3-0.6 (excellent >0.5)
- **Model Inference**: 1-3 seconds per question

## Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

Based on the paper [***Self-Prompting Large Language Models for Zero-Shot Open-Domain QA***](https://arxiv.org/abs/2212.08635) (NAACL 2024).

```bibtex
@article{li2022self,
  title={Self-prompting large language models for zero-shot open-domain qa},
  author={Li, Junlong and Wang, Jinyuan, and Zhang, Zhuosheng and Zhao, Hai},
  journal={arXiv preprint arXiv:2212.08635},
  year={2022}
}
```

## Support

- **Issues**: Open GitHub issues for bugs or feature requests
- **Documentation**: Check the `/docs` folder for detailed guides
- **Demo**: Visit the live demo at `http://localhost:5001`

---

**Star this repo if you find it helpful!**
