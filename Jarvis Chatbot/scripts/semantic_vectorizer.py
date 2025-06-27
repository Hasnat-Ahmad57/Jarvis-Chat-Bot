from scripts.load_chatterbot_data import load_chatterbot_corpus
from scripts.preprocess_text import clean_text
from sentence_transformers import SentenceTransformer
import numpy as np
import csv

def load_custom_qa(custom_csv_path):
    qa_pairs = []
    with open(custom_csv_path, 'r', encoding='utf-8', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row.get("question") and row.get("answer"):
                qa_pairs.append({
                    "question": row["question"].strip(),
                    "answer": row["answer"].strip()
                })
    return qa_pairs

def merge_and_encode(chatterbot_path, custom_csv_path="data/custom_qa.csv"):
    chatterbot_qas = load_chatterbot_corpus(chatterbot_path)
    custom_qas = load_custom_qa(custom_csv_path)
    combined_qas = custom_qas + chatterbot_qas

    print(f"✅ Combined Q&A total: {len(combined_qas)}")

    raw_questions = [q["question"] for q in combined_qas]
    cleaned_questions = [clean_text(q, remove_stopwords=False) for q in raw_questions]

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(cleaned_questions, show_progress_bar=True)

    return combined_qas, model, np.array(embeddings)
