from graphs.ToolAgent import ToolAgent

def main():
    agent = ToolAgent()

    # user_message = "What is 1234 * 5678?"
    user_message = "What is the Bitcoin price today?"
    response = agent.invoke(user_message)
    print(f"Agent: {response}")

if __name__ == "__main__":
    main()