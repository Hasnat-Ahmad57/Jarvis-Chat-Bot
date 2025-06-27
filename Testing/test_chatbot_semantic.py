# Load all Q&A and encode them
qa_data, model, question_embeddings = merge_and_encode("data/chatterbot_corpus")

print("✅ Bot is ready!\n")

# 🧪 Optional: Auto test a few known good questions
test_cases = [
    "how are you",
    "who made you",
    "do you sleep",
    "can you help me",
    "what's your name",
]

for test in test_cases:
    print(f"🧪 Test: {test}")
    print("Bot:", generate_response(test, model, question_embeddings, qa_data))
    print("")

print("💬 Ready for live chat. Ask me anything!\n")

# 🤖 Main chat loop
while True:
    user_input = input("You: ")
    if user_input.lower().strip() in ["exit", "quit"]:
        print("Bot: See you later! 👋")
        break

    response = generate_response(user_input, model, question_embeddings, qa_data)
    print("Bot:", response)

    # 📝 Optional: log conversation to file
    with open("chat_log.txt", "a", encoding="utf-8") as f:
        f.write(f"You: {user_input}\nBot: {response}\n\n")
