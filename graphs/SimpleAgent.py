# IMPORT
from typing import Annotated            # e.g. Annotated[int, "Age"]
from typing_extensions import TypedDict # define dictionaries with specific key-value types

from langchain_core.messages import HumanMessage    # langchain_core.messages defines how message payloads are structured (types of message: HumanMessage, AIMessage, SystemMessage, ...)
                                                    # A HumanMessage is a message that is passed in from a user to the model.
from langgraph.graph import StateGraph, START, END  # StateGraph: Graph classs, START: special node (start of the graph), END: special node (end of the graph)
from langgraph.graph.message import add_messages    # reducer: function that says to LangGraph how to update the msg list when a node returns a new message (sth) --> acumula en vez de reemplazar

from langchain_groq import ChatGroq
from config.settings import GROQ_API_KEY, MODEL_ID


class State(TypedDict):
    """Represents the state of the agent, including its current position in the graph and any relevant information about the environment."""
    messages: Annotated[list, add_messages]

class SimpleAgent:
    """A simple agent that interacts with a language model through a state graph, processing user messages and generating responses."""
    def __init__(self):
        self._llm = ChatGroq(
                model=MODEL_ID,
                api_key=GROQ_API_KEY,
                temperature=0.4,
                max_tokens=512,
            )
        self._graph = self._build_graph()

    def _build_graph(self):
        """Builds the state graph for the agent, defining the nodes and edges that represent the flow of information and decision-making."""
        graph = StateGraph(state_schema=State)

        # START --> llm --> END
        graph.add_node('llm_node', self._llm_node) 
        graph.add_edge(START, 'llm_node')
        graph.add_edge('llm_node', END)
        return graph.compile()  # validates graph (non-connected nodes, cycles, etc.) and returns a callable function that can be used to run the graph
    
    def _llm_node(self, state: State) -> dict:          # all nodes recieve the full state as first argument 
        """Node that interacts with the language model, taking the current state and returning an updated state with the model's response."""
        response = self._llm.invoke(state["messages"])  # returns AIMessage object with the model's response to the messages in the state: AIMessage(content="Glucose is...")
        return {"messages": [response]}                 # and return a dictionary with the updated state

    def invoke(self, user_message: str) -> str:
        """Invokes the agent with a user message and returns the model's response."""
        state = self._graph.invoke({"messages": [HumanMessage(content=user_message)]})  # run the graph with the initial state containing the user message
        return state["messages"][-1].content  # return the content of the last message in the updated state (the model's response)
        