!pip install numpy==1.25.0
!pip install gensim
!pip uninstall -y gensim numpy scipy

import gensim.downloader as api
import random
import nltk

nltk.download('punkt')

# Load pre-trained word vectors
word_vectors = api.load("glove-wiki-gigaword-100")

def get_similar_words(seed_word, top_n=5):
    """Retrieve top-N similar words for a given seed word."""
    try:
        return [word[0] for word in word_vectors.most_similar(seed_word, topn=top_n)]
    except KeyError:
        print(f"'{seed_word}' not found in vocabulary.")
        return []

def generate_paragraph(seed_word):
    """Construct a creative paragraph using the seed word and similar words."""
    similar_words = get_similar_words(seed_word)
    if not similar_words:
        return "Could not generate a paragraph. Try another seed word."

    templates = [
        f"The {seed_word} was surrounded by {similar_words[0]} and {similar_words[1]}.",
        f"People often associate {seed_word} with {similar_words[2]} and {similar_words[3]}.",
        f"In the land of {seed_word}, {similar_words[4]} was a common sight.",
        f"A story about {seed_word} would be incomplete without {similar_words[1]} and {similar_words[3]}.",
    ]

    return " ".join(random.sample(templates, len(templates)))

if __name__ == "__main__":
    seed_word = input("Enter a seed word: ").strip().lower()
    if seed_word.isalpha():
        print("\nGenerated Paragraph:\n", generate_paragraph(seed_word))
    else:
        print("Invalid input. Please enter a valid word.")
