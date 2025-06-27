import streamlit as st
from scripts.semantic_vectorizer import merge_and_encode
from scripts.semantic_response import generate_response
from scripts.preprocess_text import clean_text
import numpy as np
import csv
import os
import base64

CUSTOM_QA_PATH = "data/custom_qa.csv"

# 🖼️ Add background image from file (base64-encoded for compatibility)
def add_bg_image():
    with open("Background/bg.jpg", "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# ✅ Save new Q&A pair
def save_new_qa(question, answer):
    with open(CUSTOM_QA_PATH, "a", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "answer"])
        if os.stat(CUSTOM_QA_PATH).st_size == 0:
            writer.writeheader()
        writer.writerow({"question": question.strip(), "answer": answer.strip()})

# ✅ Load model and data
@st.cache_resource
def load_bot():
    qa_data, model, question_embeddings = merge_and_encode("data/chatterbot_corpus")
    return qa_data, model, question_embeddings

# 🟩 Setup
st.set_page_config(page_title="HasnatBot", page_icon="🤖")
add_bg_image()
st.title("🤖 Jarvis AI")

qa_data, model, question_embeddings = load_bot()

# 🧠 Initialize chat memory
if "chat" not in st.session_state:
    st.session_state.chat = []

# 🗑️ Clear chat
if st.button("🗑️ Clear Chat"):
    st.session_state.chat = []

# 📝 Input box
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("Ask something:")
    submitted = st.form_submit_button("Send")

# 🚀 Handle input
if submitted and user_input.strip():
    response = generate_response(user_input, model, question_embeddings, qa_data)
    st.session_state.chat.append(("You", user_input))
    st.session_state.chat.append(("Jarvis", response))

# 💬 Display chat in reverse (newest on top)
for sender, msg in reversed(st.session_state.chat):
    st.markdown(f"**{sender}:** {msg}")

# ✏️ Correction handler
if st.session_state.chat:
    last_response = st.session_state.chat[-1][1]
    last_input = st.session_state.chat[-2][1] if len(st.session_state.chat) >= 2 else ""

    if "I didn't understand" in last_response or "I'm not sure how to answer" in last_response:
        if st.toggle("🤔 Want to teach me this?"):
            corrected = st.text_input("✏️ Enter the correct answer below:", key="corrected")
            if st.button("✅ Save correction"):
                save_new_qa(last_input, corrected)
                qa_data.append({"question": last_input, "answer": corrected})
                cleaned = clean_text(last_input, remove_stopwords=False)
                new_embedding = model.encode([cleaned])
                question_embeddings = np.vstack([question_embeddings, new_embedding])
                st.success("🧠 Learned! You can ask again right now.")
