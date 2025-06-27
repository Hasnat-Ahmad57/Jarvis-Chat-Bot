from scripts.load_chatterbot_data import load_chatterbot_corpus

if __name__ == "__main__":
    data = load_chatterbot_corpus("../data/chatterbot_corpus")
    print(f"✅ Loaded {len(data)} Q&A pairs from ChatterBot corpus")
    print("🔍 Sample:")
    for pair in data[:5]:
        print("Q:", pair["question"])
        print("A:", pair["answer"])
        print("---")
