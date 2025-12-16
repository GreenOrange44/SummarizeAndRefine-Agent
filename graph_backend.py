import operator
import asyncio
from typing import Annotated, List, TypedDict, Union
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from tools import llm, text_splitter

# State Definition
class AgentState(TypedDict):
    text: str                         # The input text
    chunks: List[str]                 # Split parts
    chunk_summaries: List[str]        # Summaries of each chunk
    final_summary: str                # The final summary
    iteration_count: int              # To prevent infinite loops
    max_words: int                   # Max lines allowed in summary



# define the Nodes

def split_text_node(state: AgentState):
    """Node 1: Splits the input text into manageable chunks."""
    if state["final_summary"]:
        text = state["final_summary"]
    else:
        text = state["text"]
    docs = text_splitter.create_documents(text)
    chunks = [d.page_content for d in docs]
    return {"chunks": chunks}

async def summarize_chunks_parallel(state: AgentState):
    """
    Node 2: The 'Map' Step.
    Processes all chunks in parallel using llm.abatch.
    """
    # Create the prompt for parallel execution
    prompt = ChatPromptTemplate.from_template(
        "Write a concise summary of the following text, capturing all key essence and without losing important information:\n\n{text}"
    )
    chain = prompt | llm
    
    # Run all chunks in parallel (Best Practice for 'Map' in simple graphs)
    summaries = await chain.abatch([{"text": c} for c in state["chunks"]])
    
    # Extract content from AIMessage objects
    summary_texts = [s.content for s in summaries]
    return {"chunk_summaries": summary_texts}

def merge_summaries_node(state: AgentState):
    """Node 3: The 'Reduce' Step."""
    # Simple concatenation
    combined = "\n\n".join(state["chunk_summaries"])
    return {"final_summary": combined}

async def refine_summary_node(state: AgentState):
    """
    Node 4: The 'Refine' Step.
    This node is the target of the loop.
    """
    current_text = state["final_summary"]
    
    prompt_text = """
    You are a professional editor. Refine the following summary. 
    Make it more structured and flow better. Do NOT lose information.
    """
   
    prompt = ChatPromptTemplate.from_messages([
        ("system", prompt_text),
        ("human", "{text}")
    ])
    chain = prompt | llm
    
    response = await chain.ainvoke({"text": current_text})
    return {"final_summary": response.content, "iteration_count": state["iteration_count"] + 1}

#Define the Conditional Logic 

def should_continue(state: AgentState) -> str:
    """Decides whether to loop or end based on line count."""
    
    # Safety Valve: Stop after 3 iterations to prevent infinite costs
    if state["iteration_count"] >= 3:
        return "end"
    
    word_count = len(state["final_summary"].split(" "))
    
    if word_count > state["max_words"]:
        return "splitter"
    
    return "end"

# build the Graph

workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("splitter", split_text_node)
workflow.add_node("summarizer", summarize_chunks_parallel)
workflow.add_node("merger", merge_summaries_node)
workflow.add_node("refiner", refine_summary_node)

# Add Edges (Linear Flow)
workflow.add_edge(START, "splitter")
workflow.add_edge("splitter", "summarizer")
workflow.add_edge("summarizer", "merger")
workflow.add_edge("merger", "refiner")

# Add Conditional Edge (The Loop)
workflow.add_conditional_edges(
    "refiner",
    should_continue,
)

# Compile the graph
graph = workflow.compile()
