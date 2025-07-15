

#Importing libraries, Generating Dictionary, and Input Words
"""

import nltk
import random
from nltk.corpus import words
nltk.download('words')

# ✅ Grocery and real words
grocery_words = [
    'milk', 'bread', 'cheese', 'butter', 'yogurt', 'banana', 'apple', 'orange', 'onion', 'potato',
    'carrot', 'tomato', 'rice', 'wheat', 'pasta', 'chocolate', 'biscuits', 'juice', 'water', 'soda',
    'egg', 'flour', 'salt', 'sugar', 'oil', 'tea', 'coffee', 'soap', 'shampoo', 'toothpaste',
    'detergent', 'spices', 'lentils', 'beans', 'peas', 'spinach', 'lettuce', 'cucumber', 'corn'
]

daily_words = [
    'phone', 'pen', 'book', 'chair', 'table', 'bag', 'glass', 'fan', 'light', 'window', 'bottle',
    'keyboard', 'mouse', 'school', 'laptop', 'notebook', 'charger', 'door', 'bed', 'mirror'
]

# Get extra realistic words from NLTK's word corpus
nltk_words = list(set(words.words()))
additional_words = random.sample(nltk_words, 5000 - len(grocery_words + daily_words))

# Final Dictionary (5K)
final_dictionary = list(set(grocery_words + daily_words + additional_words))
random.shuffle(final_dictionary)

with open("dictionary.txt", "w") as f:
    for word in final_dictionary:
        f.write(word.lower() + "\n")

# Function to introduce typo
def misspell(word):
    if len(word) < 3: return word
    typo = random.choice(['swap', 'delete', 'replace', 'insert'])
    chars = list(word)
    idx = random.randint(0, len(chars) - 2)

    if typo == 'swap' and len(chars) > 1:
        chars[idx], chars[idx+1] = chars[idx+1], chars[idx]
    elif typo == 'delete':
        del chars[idx]
    elif typo == 'insert':
        chars.insert(idx, random.choice('abcdefghijklmnopqrstuvwxyz'))
    elif typo == 'replace':
        chars[idx] = random.choice('abcdefghijklmnopqrstuvwxyz')

    return ''.join(chars)

# Generate 10K misspelled inputs from 5K dictionary
with open("input.txt", "w") as finput, open("expected_output.txt", "w") as fcorrect:
    for _ in range(10000):
        correct_word = random.choice(final_dictionary)
        misspelled = misspell(correct_word)
        finput.write(misspelled + "\n")
        fcorrect.write(correct_word + "\n")

print("✅ Generated dictionary.txt (5000 words), input.txt (10000 words), expected_output.txt")

"""# **Set Up, Installation and Preprocessing**"""

!pip install -q textdistance fuzzy
import nltk
nltk.download('words')

import pandas as pd
import textdistance
import fuzzy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Double Metaphone encoder
dmeta = fuzzy.DMetaphone()

"""# **Helper Functions**"""

def get_phonetic(word):
    meta = dmeta(word)
    return meta[0] if meta[0] else b''

def load_inputs(path='input.txt'):
    with open(path, 'r') as f:
        return [line.strip().lower() for line in f if line.strip()]

def load_expected(path='expected_output.txt'):
    with open(path, 'r') as f:
        return [line.strip().lower() for line in f if line.strip()]

def load_dictionary(path='dictionary.txt'):
    with open(path, 'r') as f:
        return [line.strip().lower() for line in f if line.strip()]

def setup_vectorizer(dictionary_words):
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4))
    dictionary_vectors = vectorizer.fit_transform(dictionary_words)
    return vectorizer, dictionary_vectors

"""# **Spell Correction Logic (Top-k Hybrid Match)**


"""

from tqdm import tqdm
import numpy as np
import textdistance

def correct_top_k(word, vectorizer, dictionary_vectors, dictionary_words, k=3, filter_top_n=50):
    input_vector = vectorizer.transform([word])
    tfidf_scores = cosine_similarity(input_vector, dictionary_vectors).flatten()

    # Pick top-N TF-IDF candidates only
    top_indices = tfidf_scores.argsort()[-filter_top_n:][::-1]

    word_meta = get_phonetic(word)
    word_meta = word_meta.decode("utf-8") if word_meta else ""

    candidates = []
    for i in top_indices:
        candidate = dictionary_words[i]
        tfidf = tfidf_scores[i]
        edit = textdistance.levenshtein.normalized_similarity(word, candidate)
        candidate_meta = get_phonetic(candidate)
        candidate_meta = candidate_meta.decode("utf-8") if candidate_meta else ""
        phonetic = 1.0 if word_meta and word_meta == candidate_meta else 0.0
        score = 0.5 * tfidf + 0.3 * edit + 0.2 * phonetic
        candidates.append((candidate, score))

    return sorted(candidates, key=lambda x: x[1], reverse=True)[:k]

"""# **Main Logic / Engine + Accuracy Evaluation**"""

from tqdm import tqdm

def run_spell_corrector():
    input_words = load_inputs()
    expected_outputs = load_expected()
    dictionary_words = load_dictionary()

    vectorizer, dictionary_vectors = setup_vectorizer(dictionary_words)

    results = []
    wrong_predictions = []
    top1_correct = 0
    top3_correct = 0
    total = len(input_words)

    for i, input_word in tqdm(enumerate(input_words), total=len(input_words)):
        expected = expected_outputs[i]
        top_k = correct_top_k(input_word, vectorizer, dictionary_vectors, dictionary_words, k=3)
        top1 = top_k[0][0]
        top3_words = [w for w, _ in top_k]

        is_top1 = (top1 == expected)
        is_top3 = (expected in top3_words)

        if is_top1:
            top1_correct += 1
        if is_top3:
            top3_correct += 1

        row = {
            "Input (Misspelled)": input_word,
            "Expected Output": expected,
            "Predicted Output (Top-1)": top1,
            "Top-3 Predictions": ', '.join(top3_words),
            "Correct?": "✅" if is_top1 else "❌"
        }

        results.append(row)
        if not is_top1:
            wrong_predictions.append(row)

    pd.DataFrame(results).to_csv("correction_results.csv", index=False)
    pd.DataFrame(wrong_predictions).to_csv("wrong_predictions.csv", index=False)

    acc1 = top1_correct / total * 100
    acc3 = top3_correct / total * 100
    print("📊 Accuracy Summary")
    print(f"✅ Top-1 Accuracy: {acc1:.2f}% ({top1_correct}/{total})")
    print(f"✅ Top-3 Accuracy: {acc3:.2f}% ({top3_correct}/{total})")
    print("📁 Saved:")
    print("   - correction_results.csv")
    print("   - wrong_predictions.csv")

# Run it
run_spell_corrector()
