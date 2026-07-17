from graphs.SimpleAgent import SimpleAgent

def main():
    agent = SimpleAgent()

    user_message = "What is the role of glucose in the human body?"
    response = agent.invoke(user_message)
    print(f"Agent: {response}")

if __name__ == "__main__":
    main()