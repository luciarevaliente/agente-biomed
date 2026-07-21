# IMPORT
from typing import Annotated            # e.g. Annotated[int, "Age"]
from typing_extensions import TypedDict # define dictionaries with specific key-value types

from langchain_core.messages import HumanMessage    # langchain_core.messages defines how message payloads are structured (types of message: HumanMessage, AIMessage, SystemMessage, ...)
                                                    # A HumanMessage is a message that is passed in from a user to the model.
from langgraph.graph import StateGraph, START, END  # StateGraph: Graph classs, START: special node (start of the graph), END: special node (end of the graph)
from langgraph.graph.message import add_messages    # reducer: function that says to LangGraph how to update the msg list when a node returns a new message (sth) --> acumula en vez de reemplazar

from langchain_groq import ChatGroq
from config.settings import GROQ_API_KEY, MODEL_ID

# New imports for ToolAgent
from langchain_core.tools import tool                     # decorator to define a tool function that can be used by the agent --> etiqueta con nombre, descripción y tipos --> LLM lee etiquetas para decidir qué tool usar
from langgraph.prebuilt import ToolNode, tools_condition  # ToolNode: prebuilt node that allows to use a tool in the graph; tools_condition: prebuilt condition that allows to decide which tool to use based on the user message

from ddgs import DDGS  # DuckDuckGo search API wrapper

class State(TypedDict):
    """Represents the state of the agent, including its current position in the graph and any relevant information about the environment."""
    messages: Annotated[list, add_messages]

# Tools have 3 parts: name, description, type of params. LLM use this 3 to decide when and how to use it.
@tool
def search(query:str) -> str:
    """Search th web for information about a topic.
    Use this when you need current or specific information you don't know."""
    print(f"[search] searching for: {query}") 
    with DDGS() as ddgs: # open a DuckDuckGo search session
        results = ddgs.text(query, max_results=3) # search for the query and return the top 3 results as a list of dictionaries
    return "\n".join([r["body"] for r in results]) # return the body of the results as a single string, separated by newlines

@tool
def calculator(expression:str) -> str:  # a mejorar en las siguientes fases :)
    """Evaluate a mathematical expression
    Use this for arithmetic calculation. Example: '2 + 2', '3 * 5', '100 / 4', 'sqrt(16)'."""
    print(f"[calculator] evaluating: {expression}") 
    try:
        result = eval(expression)  # evaluate the expression using Python's eval function
        return str(result)         # return the result as a string
    except Exception as e:
        return f"Error evaluating expression: {e}"  # return an error message if the evaluation fails
    

class ToolAgent:
    """An agent that interacts with a language model and can use tools through a state graph, processing user messages and generating responses."""
    def __init__(self):
        
        self._tools = [search, calculator]  # list of tools that the agent can use
        self._llm = ChatGroq(
            model=MODEL_ID,
            api_key=GROQ_API_KEY,
            temperature=0.4,
            max_tokens=512,
        ).bind_tools(self._tools) # conversational wrapper; without bind_tools LLM doesn't know that exist tools
        self._tool_node = ToolNode(self._tools)  # prebuilt node that will use the tools when the LLM asks for it
        self._graph = self._build_graph()

    def _build_graph(self):
        """Builds the state graph for the agent, defining the nodes and edges that represent the flow of information and decision-making."""
        graph = StateGraph(state_schema=State)

        # nodes
        graph.add_node('llm_node', self._llm_node) 
        graph.add_node('tool_node', self._tool_node) 

        # edges
        graph.add_edge(START, 'llm_node')
        graph.add_conditional_edges('llm_node', tools_condition, {'tools': 'tool_node', "__end__": END})  # if LLM asks for a tool, go to tool_node; else go to END
        graph.add_edge('tool_node', 'llm_node')  

        return graph.compile()  # validates graph (non-connected nodes, cycles, etc.) and returns a callable function that can be used to run the graph
    
    def _llm_node(self, state: State) -> dict:          # all nodes recieve the full state as first argument 
        """Node that interacts with the language model, taking the current state and returning an updated state with the model's response."""
        response = self._llm.invoke(state["messages"])  # returns AIMessage object with the model's response to the messages in the state: AIMessage(content="Glucose is...")
        return {"messages": [response]}                 # and return a dictionary with the updated state

    def invoke(self, user_message: str) -> str:
        """Invokes the agent with a user message and returns the model's response."""
        state = self._graph.invoke({"messages": [HumanMessage(content=user_message)]})  # run the graph with the initial state containing the user message
        return state["messages"][-1].content  # return the content of the last message in the updated state (the model's response)


