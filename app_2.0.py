import streamlit as st
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from typing import TypedDict, List
from dotenv import load_dotenv

# ==============================================================================
# --- 1. LANGGRAPH RAG LOGIC (UNCHANGED) ---
# This section contains the backend logic and is not modified.
# ==============================================================================

load_dotenv()

# --- Define the State for our Graph ---
class GraphState(TypedDict):
    question: str
    generation: str
    chat_history: List[BaseMessage]
    documents: List[str]
    sources: List[str]

# --- Setup Database and LLM ---
CHROMA_PATH = "chroma"

embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
retriever = db.as_retriever(k=3)
# Note: Using the standard name for the latest flash model for best performance
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", streaming=True)

# --- Define the Graph Nodes ---
def rewrite_query(state: GraphState):
    print("---REWRITING QUERY---")
    question = state["question"]
    chat_history = state["chat_history"]
    if not chat_history: return {"question": question}
    
    rewrite_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an expert at rephrasing a follow-up question to be a standalone question, using the context of a chat history."),
            ("placeholder", "{chat_history}"),
            ("human", "Based on the chat history, rephrase the following follow-up question into a standalone question.\n"
                      "Your ONLY job is to rephrase the question. DO NOT answer it.\n"
                      "Follow-up Question: {question}"),
        ]
    )
    rewriter = rewrite_prompt | llm | StrOutputParser()
    rewritten_question = rewriter.invoke({"chat_history": chat_history, "question": question})
    print(f"Rewritten question: {rewritten_question}")
    return {"question": rewritten_question}

def retrieve_documents(state: GraphState):
    print("---RETRIEVING DOCUMENTS---")
    question = state["question"]
    documents = retriever.invoke(question)
    doc_contents = [doc.page_content for doc in documents]
    sources = [doc.metadata.get("source", "N/A") for doc in documents]
    return {"documents": doc_contents, "question": question, "sources": sources}

def generate_response_rag(state: GraphState):
    print("---GENERATING RAG RESPONSE---")
    question = state["question"]
    documents = state["documents"]
    chat_history = state["chat_history"]
    context_text = "\n\n---\n\n".join(documents)
    prompt_template = ChatPromptTemplate.from_messages([("system", "You are a helpful assistant. Answer the user's question based on the following context and the conversation history.\n\nContext:\n{context}"), ("placeholder", "{chat_history}"), ("human", "{question}")])
    chain = prompt_template | llm | StrOutputParser()
    # The `stream` method is implicitly used by LangGraph when the graph is streamed.
    return {"generation": chain.stream({"context": context_text, "chat_history": chat_history, "question": question})}

def generate_conversational_response(state: GraphState):
    print("---GENERATING CONVERSATIONAL RESPONSE---")
    question = state["question"]
    chat_history = state["chat_history"]
    prompt_template = ChatPromptTemplate.from_messages([("system", "You are a helpful and friendly chatbot. Respond to the user's message conversationally."), ("placeholder", "{chat_history}"), ("human", "{question}")])
    chain = prompt_template | llm | StrOutputParser()
    # The `stream` method is implicitly used by LangGraph when the graph is streamed.
    return {"generation": chain.stream({"chat_history": chat_history, "question": question}), "sources": []}

def route_question(state: GraphState):
    print("---ROUTING QUESTION---")
    question = state["question"]
    prompt_template = ChatPromptTemplate.from_template("""Given the user's question below, classify it as either "rag" or "conversational".
    Do not respond with more than one word.

    - "rag": For questions that require specific information from a knowledge base.
    - "conversational": For greetings, thank yous, or other conversational filler.

    Question: {question}
    Classification:""")
    router_chain = prompt_template | llm | StrOutputParser()
    result = router_chain.invoke({"question": question})
    print(f"Route: {result}")
    if "conversational" in result.lower():
        return "conversational"
    else:
        return "rag"

# --- Build the Graph ---
workflow = StateGraph(GraphState)

workflow.add_node("rewrite_query", rewrite_query)
workflow.add_node("retrieve_documents", retrieve_documents)
workflow.add_node("generate_rag_response", generate_response_rag)
workflow.add_node("generate_conversational_response", generate_conversational_response)

workflow.set_conditional_entry_point(
    route_question,
    {"rag": "rewrite_query", "conversational": "generate_conversational_response"},
)

workflow.add_edge("rewrite_query", "retrieve_documents")
workflow.add_edge("retrieve_documents", "generate_rag_response")
workflow.add_edge("generate_rag_response", END)
workflow.add_edge("generate_conversational_response", END)

# The compiled LangGraph app
app = workflow.compile()


# ==============================================================================
# --- 2. STREAMLIT UI (NOW WITH STREAMING) ---
# ==============================================================================

# --- Page Configuration ---
st.set_page_config(
    page_title="Chat with the Constitution",
    page_icon="🇵🇰",
    layout="centered"
)

# --- Sidebar ---
with st.sidebar:
    st.title("🇵🇰 Chat with the Constitution")
    st.markdown("""
    This app is a chatbot that can answer questions about the Constitution of the Islamic Republic of Pakistan. 
    It uses a Retrieval-Augmented Generation (RAG) pipeline to provide accurate answers.
    """)
    if st.button("Clear Chat History", use_container_width=True, type="primary"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()
    st.markdown("---")
    st.markdown("**Source:** [Constitution of Pakistan (PDF)](https://www.pakp.gov.pk/wp-content/uploads/2024/07/Constitution.pdf)")

# --- Session State Management ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    
# --- Main Content ---

# if not st.session_state.messages:
st.title("Chat with the Constitution of Pakistan 🇵🇰")
st.markdown("I can answer questions about the Constitution of Pakistan. Try asking me something!")

# --- Display Chat History ---
for message in st.session_state.messages:
    avatar = "🧑‍💻" if message["role"] == "user" else "🇵🇰"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# --- Handle User Input ---
if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🇵🇰"):
        # MODIFICATION: Create a generator for streaming the response
        def stream_response_generator():
            inputs = {
                "question": prompt,
                "chat_history": st.session_state.chat_history
            }
            # Use app.stream() which yields events for each step. We parse these events.
            for event in app.stream(inputs):
                # The event dictionary's key is the name of the node that just ran
                node_name = list(event.keys())[0]
                
                # Check if the node is one of our generation nodes
                if node_name in ["generate_rag_response", "generate_conversational_response"]:
                    # The 'generation' value from these nodes is now a stream itself
                    generation_stream = event[node_name]['generation']
                    for chunk in generation_stream:
                        yield chunk
        
        # Use st.write_stream to render the response in real-time
        full_response = st.write_stream(stream_response_generator)
        
        # Once the stream is complete, save the full response to the session state
        st.session_state.messages.append({
            "role": "assistant", 
            "content": full_response
        })

        st.session_state.chat_history.extend([
            HumanMessage(content=prompt),
            AIMessage(content=full_response)
        ])
