from transformers import pipeline
from nltk.tokenize import word_tokenize
import gensim.downloader as api
import nltk, string

nltk.download('punkt_tab')
word_vectors = api.load("glove-wiki-gigaword-100")
generator = pipeline("text-generation", model="gpt2")

def enrich_prompt(prompt, keyword, topn=1):
    # Tokenize the prompt into individual words
    words = word_tokenize(prompt)

    enriched_words = []

    for word in words:
        cleaned_word = word.lower().strip(string.punctuation)

        if cleaned_word == keyword.lower():
            if cleaned_word in word_vectors:
                similar_words = word_vectors.most_similar(cleaned_word, topn=topn)
                replacement_word = similar_words[0][0]  # Take the top similar word
                enriched_words.append(replacement_word)
                continue 
            else:
                enriched_words.append(word)
        else:
            enriched_words.append(word)
    enriched_prompt = " ".join(enriched_words)
    return enriched_prompt

def get_response(prompt):
    return generator(prompt, max_length=100, num_return_sequences=1)[0]['generated_text']

# Input
original_prompt = "Who is king."
keyword = "king"

# Enrichment and response
enriched_prompt = enrich_prompt(original_prompt, keyword)
original_response = get_response(original_prompt)
enriched_response = get_response(enriched_prompt)

# Output comparison
print(f"\nOriginal Prompt: {original_prompt}")
print(f"Enriched Prompt: {enriched_prompt}")
print(f"\nOriginal Response:\n{original_response}")
print(f"\nEnriched Response:\n{enriched_response}")
