# Transformers Lectures 🎓

A comprehensive collection of Jupyter notebooks and educational materials for teaching neural networks, sequence modeling, and the evolution towards modern transformers.

## 📚 Course Overview

This repository contains pedagogical materials for understanding the foundational concepts that led to the development of transformers, including:

- **Neural Networks as Matrix Operations** - Understanding the mathematical foundations
- **Sequence Modeling** - RNNs, LSTMs, and their applications
- **Autoencoders** - Dimensionality reduction and representation learning
- **Word Embeddings** - From Word2Vec to modern embeddings
- **Historical Evolution** - From Hopfield networks to attention mechanisms

## 🗂️ Repository Structure

```
Session_1/
├── 1_spiral_classification.ipynb          # Neural network fundamentals
├── 2_seq_classification.ipynb             # RNN/LSTM for sequence tasks
├── 3_autoencoder.ipynb                    # Autoencoder implementation
├── 4_intro_word_embeddings.ipynb          # Word embeddings and sentiment analysis
├── res/                                   # Supporting resources
│   ├── plot_lib.py                       # Custom plotting utilities
│   ├── sequential_tasks.py                # Sequence modeling utilities
│   └── *.png                             # Visual assets
├── From Foundations to Transformers.html   # Lecture notes
├── Hopfield Networks: Associative Memory.html
└── hopfield-1982-neural-networks-and-physical-systems-with-emergent-collective-computational-abilities.pdf
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Jupyter Notebook/Lab
- CUDA (optional, for GPU acceleration)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/akashjorss/Transformers_Lectures.git
   cd Transformers_Lectures
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch Jupyter:**
   ```bash
   jupyter lab
   # or
   jupyter notebook
   ```

## 📖 Notebooks Overview

### 1. Spiral Classification (`1_spiral_classification.ipynb`)
- **Learning Objectives:** Understand neural networks as matrix operations
- **Key Concepts:** Linear layers, activation functions, backpropagation
- **Techniques:** Multi-layer perceptrons, non-linear decision boundaries
- **Visualization:** 2D spiral data classification

### 2. Sequence Classification (`2_seq_classification.ipynb`)
- **Learning Objectives:** Master sequence modeling with RNNs and LSTMs
- **Key Concepts:** Recurrent networks, vanishing gradients, LSTM architecture
- **Techniques:** Many-to-one classification, temporal order tasks
- **Applications:** Text classification, time series analysis

### 3. Autoencoder (`3_autoencoder.ipynb`)
- **Learning Objectives:** Learn representation learning and dimensionality reduction
- **Key Concepts:** Encoder-decoder architecture, reconstruction loss
- **Techniques:** Standard vs. denoising autoencoders, image inpainting
- **Applications:** Data compression, anomaly detection

### 4. Word Embeddings (`4_intro_word_embeddings.ipynb`)
- **Learning Objectives:** Understand word representations and semantic arithmetic
- **Key Concepts:** Embedding layers, pre-trained embeddings, semantic similarity
- **Techniques:** Word2Vec, GloVe, sentiment analysis
- **Applications:** Natural language processing, semantic search

## 🎯 Learning Path

### For Beginners:
1. Start with **Spiral Classification** to understand basic neural networks
2. Move to **Sequence Classification** for temporal modeling
3. Explore **Autoencoders** for unsupervised learning
4. Finish with **Word Embeddings** for NLP foundations

### For Advanced Students:
- Focus on the mathematical connections between concepts
- Implement extensions and modifications
- Explore the historical papers included in the repository

## 📚 Additional Resources

### Lecture Notes
- **From Foundations to Transformers.html** - Comprehensive overview
- **Hopfield Networks: Associative Memory.html** - Historical context

### Research Papers
- **hopfield-1982-neural-networks-and-physical-systems-with-emergent-collective-computational-abilities.pdf** - Original Hopfield network paper

## 🛠️ Technical Details

### Dependencies
- **PyTorch** - Deep learning framework
- **Matplotlib/Seaborn** - Visualization
- **Scikit-learn** - Machine learning utilities
- **TensorFlow** - Dataset loading (IMDB)
- **OpenCV** - Image processing
- **Gensim** - Word embeddings

### Hardware Requirements
- **CPU:** Modern multi-core processor
- **RAM:** 8GB+ recommended
- **GPU:** Optional, but recommended for faster training
- **Storage:** 2GB for datasets and models

## 🤝 Contributing

This is an educational repository. Contributions are welcome in the form of:
- Bug fixes and improvements
- Additional examples and exercises
- Enhanced visualizations
- Documentation improvements

## 📚 Acknowledgments

This repository incorporates educational materials from the following sources:

- **[NYU Deep Learning Spring 2021](https://atcold.github.io/NYU-DLSP21/)** - Course materials by Yann LeCun & Alfredo Canziani
- **[TensorFlow Word Embeddings Tutorial](https://colab.research.google.com/github/securetorobert/docs/blob/master/site/en/tutorials/keras/intro_word_embeddings.ipynb)** - Original word embeddings tutorial adapted for PyTorch

We gratefully acknowledge the original authors and maintainers of these educational resources.

## 📄 License

This project is intended for educational purposes. Please respect the licenses of any third-party datasets and models used.

## 👨‍🏫 Instructor Notes

### Teaching Tips
1. **Start with visualization** - Use the plotting utilities to show concepts
2. **Encourage experimentation** - Modify hyperparameters and architectures
3. **Connect to history** - Reference the included research papers
4. **Focus on intuition** - Explain the "why" behind each technique

### Assessment Ideas
- Implement extensions to the provided notebooks
- Compare different architectures on the same task
- Write explanations of the mathematical concepts
- Create visualizations of the learning process

## 🔗 Related Resources

- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer paper
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) - Visual explanation
- [Hugging Face Course](https://huggingface.co/course) - Modern NLP with transformers

---

**Happy Learning! 🚀**

*For questions or issues, please open an issue in this repository.*
