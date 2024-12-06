from main import GenerativeAIChatbot

example_conversations = [
    {"input": "hello", "output": "hi there"},
    {"input": "how are you", "output": "i'm doing great, thanks"},
    {"input": "what's your name", "output": "i'm an AI chatbot"},
    {"input": "tell me a joke", "output": "why did the AI cross the road? To optimize the path!"}
]

chatbot = GenerativeAIChatbot()
chatbot.load_model()

while True:
    # Kiểm tra chatbot
    test_input = input("You: ")
    response = chatbot.generate_response(test_input)
    print(f"Input: {test_input}")
    print(f"Response: {response}")
