from scripts.semantic_vectorizer import merge_and_encode
from scripts.semantic_response import generate_response
from scripts.preprocess_text import clean_text
import csv
import os
import numpy as np

CUSTOM_QA_PATH = "data/custom_qa.csv"

def save_new_qa(question, answer):
    """Append a new Q&A pair to the custom CSV file."""
    with open(CUSTOM_QA_PATH, "a", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "answer"])
        if os.stat(CUSTOM_QA_PATH).st_size == 0:
            writer.writeheader()
        writer.writerow({"question": question.strip(), "answer": answer.strip()})
    print("✅ New answer saved to custom_qa.csv!")

def main():
    print("🤖 Welcome to HasnatBot (Self-Learning Edition)!")
    print("⏳ Loading knowledge base...\n")

    # Load data and encode questions
    qa_data, model, question_embeddings = merge_and_encode("data/chatterbot_corpus")

    print("✅ Ready! Type your message. Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower().strip() in ["exit", "quit"]:
            print("Bot: Goodbye! 👋")
            break

        # Response + similarity
        response = generate_response(user_input, model, question_embeddings, qa_data)
        print("Bot:", response)

        # Ask if response was helpful (always)
        fix = input("🧠 Was that answer correct? (y/n): ").strip().lower()
        if fix == 'n':
            new_answer = input("🔁 Enter the correct answer for this question: ").strip()
            review = input(f"💾 Save this answer? (y/n): ").strip().lower()

            while review == 'n':
                new_answer = input("✏️ Re-enter your updated answer: ").strip()
                review = input(f"💾 Save this answer? (y/n): ").strip().lower()

            if review == 'y':
                # Save corrected Q&A (based on original input)
                save_new_qa(user_input, new_answer)

                # Update memory
                new_pair = {"question": user_input.strip(), "answer": new_answer.strip()}
                qa_data.append(new_pair)

                # Re-encode and append
                cleaned = clean_text(user_input.strip(), remove_stopwords=False)
                new_embedding = model.encode([cleaned])
                question_embeddings = np.vstack([question_embeddings, new_embedding])
                print("✅ Correction saved and embedded!\n")

if __name__ == "__main__":
    main()
