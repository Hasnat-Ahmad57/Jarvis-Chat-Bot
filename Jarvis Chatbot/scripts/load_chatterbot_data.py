import os
import yaml
import glob

def load_chatterbot_corpus(corpus_dir):
    qa_pairs = []

    for filepath in glob.glob(os.path.join(corpus_dir, '*.yml')):
        with open(filepath, 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)
            if not data or 'conversations' not in data:
                continue

            for convo in data['conversations']:
                if isinstance(convo, list) and len(convo) >= 2:
                    question = str(convo[0]).strip()
                    answer = str(convo[1]).strip()
                    qa_pairs.append({"question": question, "answer": answer})

    return qa_pairs
