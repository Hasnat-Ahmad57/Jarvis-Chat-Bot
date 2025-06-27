from scripts.preprocess_text import clean_text
from sklearn.metrics.pairwise import cosine_similarity
from sympy import symbols, Eq, solve
import re
import requests

# 🔑 Your SerpAPI key
SERP_API_KEY = "c79809b5de01c56f4a8eda8770671de990008e60be7b4f9908d030c66331528a"

# 🌐 Fallback search using Google via SerpAPI
def search_google(query):
    try:
        params = {
            "engine": "google",
            "q": query,
            "api_key": SERP_API_KEY
        }
        response = requests.get("https://serpapi.com/search", params=params)
        data = response.json()

        # Try to return answer box or snippet
        if "answer_box" in data:
            return (
                data["answer_box"].get("snippet")
                or data["answer_box"].get("answer")
                or "Found something, but I couldn't extract a clear answer."
            )
        elif "organic_results" in data and data["organic_results"]:
            return data["organic_results"][0].get("snippet", "I found something, but couldn't extract a snippet.")
        else:
            return "I searched online, but couldn’t find a clear answer either."

    except Exception as e:
        return f"Error searching Google: {str(e)}"

# 🔁 Main response generator
def generate_response(user_input, model, embeddings, qa_pairs):
    cleaned_input = user_input.strip()

    # ✅ Handle implicit multiplication: 2(3+4) → 2*(3+4)
    cleaned_input = re.sub(r"(\d)\s*\(", r"\1*(", cleaned_input)

    # ✅ Basic arithmetic expression
    if re.fullmatch(r"[0-9\s\.\+\-\*/\(\)]+", cleaned_input.replace(" ", "")):
        try:
            result = eval(cleaned_input)
            return f"The answer is: {result}"
        except Exception:
            pass  # fallback

    # ✅ Algebraic expression
    if "=" in cleaned_input:
        try:
            x = symbols("x")
            left, right = cleaned_input.split("=")
            equation = Eq(eval(left.strip()), eval(right.strip()))
            solution = solve(equation)
            return f"The solution is: x = {solution[0]}" if solution else "No solution found."
        except Exception:
            pass

    # ✅ Semantic similarity match
    cleaned = clean_text(cleaned_input, remove_stopwords=False)
    if not cleaned:
        return "I didn't understand that. Could you please rephrase?"

    user_embedding = model.encode([cleaned])
    scores = cosine_similarity(user_embedding, embeddings)[0]

    top_idx = scores.argmax()
    confidence = scores[top_idx]

    # 🔍 Logs
    print(f"🧪 Confidence score: {confidence:.4f}")
    print("📄 Cleaned input:", cleaned)
    print("📄 Matched cleaned:", clean_text(qa_pairs[top_idx]['question'], remove_stopwords=False))
    print("🧠 Matched question:", qa_pairs[top_idx]["question"])
    print("💬 Full Answer:", qa_pairs[top_idx]["answer"])
    print("—")

    if confidence < 0.50:
        print("🌐 Falling back to Google search...")
        return search_google(user_input)

    return qa_pairs[top_idx]["answer"]
