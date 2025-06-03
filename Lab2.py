!pip uninstall -y gensim numpy scipy
!pip install numpy==1.25.0
!pip install gensim

import gensim.downloader as api
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

# Load the GloVe model
model = api.load("glove-wiki-gigaword-50")

# Function to visualize word embeddings using PCA
def visualize_embeddings(words):
    valid_words = [word for word in words if word in model.key_to_index]

    if not valid_words:
        print("No valid words found in the model.")
        return

    vectors = np.array([model[word] for word in valid_words])
    pca = PCA(n_components=2)
    reduced_vectors = pca.fit_transform(vectors)

    plt.figure(figsize=(10, 6))
    for i, word in enumerate(valid_words):
        plt.scatter(reduced_vectors[i, 0], reduced_vectors[i, 1])
        plt.text(reduced_vectors[i, 0] + 0.02, reduced_vectors[i, 1] + 0.02, word, fontsize=12)

    plt.title("Word Embedding Visualization using PCA")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.show()

# Function to find semantically similar words
def find_similar_words(word, topn=5):
    similar_words = model.most_similar(word, topn=topn)
    return [similar_word for similar_word, _ in similar_words]

# Example words for visualization and similarity analysis
words_to_visualize = ["AI", "robot", "data", "computer", "algorithm", "technology", "science", "innovation", "automation", "machine"]

# Visualize embeddings and find similar words for user input
visualize_embeddings(words_to_visualize)
word = input("Enter a word to find similar words: ")
similar_words = find_similar_words(word)
print(f"Words similar to '{word}': {', '.join(similar_words)}")
