import random
responses = {
    "hello": ["Hi there! How can I help you?", "Hello! Nice to meet you.", "Hey! How’s it going?"],
    "hi": ["Hello!", "Hi there!", "Hey!"],
    "how are you": ["I'm just a bot, but I’m doing great!", "I’m fine, thanks for asking!", "Doing well! How about you?"],
    "weather": ["I can’t check live weather yet, but I love sunny days!", "Weather is always nice in my virtual world!", "I hope it’s sunny where you are!"],
    "bye": ["Goodbye! Have a great day!", "See you later!", "Bye! Take care!"]
}
print("Welcome to SimpleChatBot! Type 'exit' to quit.")

while True:
    user_input = input("You: ").lower()  

    if user_input == "exit":
        print("Bot: Goodbye! Chat with you later.")
        break

    response_found = False

  
    for key in responses:
        if key in user_input:
            print("Bot:", random.choice(responses[key])) 
            response_found = True
            break

    if not response_found:
        print("Bot: Sorry, I don’t understand that yet.")
