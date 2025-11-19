def simple_chatbot(user_input):
    # Convert the user input to lowercase for case-insensitive matching
    user_input = user_input.lower()

    # Define some simple rules and responses
    if "hello" in user_input:
        return "Hi there! How can I help you today?"
    elif "how are you" in user_input:
        return "I'm just a computer program, but thanks for asking!"
    elif "goodbye" in user_input or "bye" in user_input:
        return "Goodbye! Have a great day!"
    elif "age" in user_input:
        return "I don't have an age. I'm a chatbot."
    else:
        return "I'm sorry, I don't understand. Can you please rephrase or ask a different question?"

# Simple loop to keep the chatbot running
while True:
    # Get user input
    user_input = input("You: ")

    # Exit the loop if the user types "exit"
    if user_input.lower() == "exit":
        print("Chatbot: Goodbye!")
        break

    # Get and print the chatbot's response
    response = simple_chatbot(user_input)
    print("Chatbot:", response)
