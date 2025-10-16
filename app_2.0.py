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
import os

# --- 1. Load Environment Variables ---
load_dotenv()

# --- 2. Define the State for our Graph ---
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: The user's question.
        generation: The LLM's generated response.
        chat_history: The list of messages in the conversation.
        documents: A list of retrieved document contents.
        sources: A list of sources for the retrieved documents.
    """
    question: str
    generation: str
    chat_history: List[BaseMessage]
    documents: List[str]
    sources: List[str]

# --- 3. Cached Function to Load Models and Build Graph ---
@st.cache_resource
def load_graph():
    """
    Loads the embedding model, vector database, LLM, and builds the LangGraph workflow.
    This function is cached to avoid reloading models on every interaction.
    """
    # --- Initialize Models ---
    CHROMA_PATH = "chroma"
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    retriever = db.as_retriever(k=3)
    # Use a powerful and recent model for better reasoning
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.2)

    # --- Define Graph Nodes ---

    def rewrite_query(state: GraphState):
        """Rewrites the user's question to be a standalone question based on chat history."""
        print("---Executing Node: rewrite_query---")
        question = state["question"]
        chat_history = state["chat_history"]

        if not chat_history:
            return {"question": question}

        rewrite_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert at rephrasing a follow-up question to be a standalone question, using the context of a chat history."),
            ("placeholder", "{chat_history}"),
            ("human", "Based on the chat history, rephrase the following follow-up question into a standalone question.\nYour ONLY job is to rephrase the question. DO NOT answer it.\nFollow-up Question: {question}"),
        ])
        rewriter = rewrite_prompt | llm | StrOutputParser()
        rewritten_question = rewriter.invoke({"chat_history": chat_history, "question": question})
        print(f"Rewritten Question: {rewritten_question}")
        return {"question": rewritten_question}

    def retrieve_documents(state: GraphState):
        """Retrieves documents from the vector store based on the question."""
        print("---Executing Node: retrieve_documents---")
        question = state["question"]
        documents = retriever.invoke(question)
        doc_contents = [doc.page_content for doc in documents]
        sources = [doc.metadata.get("source", "N/A") for doc in documents]
        return {"documents": doc_contents, "sources": sources}

    def generate_response_rag(state: GraphState):
        """Generates a response using the retrieved documents and chat history."""
        print("---Executing Node: generate_response_rag---")
        question = state["question"]
        documents = state["documents"]
        chat_history = state["chat_history"]
        context_text = "\n\n---\n\n".join(documents)

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. Answer the user's question based on the following context and the conversation history. If the context does not contain the answer, say that the information is not available in the provided document.\n\nContext:\n{context}"),
            ("placeholder", "{chat_history}"),
            ("human", "{question}")
        ])
        chain = prompt_template | llm | StrOutputParser()
        response = chain.invoke({"context": context_text, "chat_history": chat_history, "question": question})
        return {"generation": response}

    def generate_conversational_response(state: GraphState):
        """Generates a conversational response when no retrieval is needed."""
        print("---Executing Node: generate_conversational_response---")
        question = state["question"]
        chat_history = state["chat_history"]

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful and friendly chatbot. Respond to the user's message conversationally."),
            ("placeholder", "{chat_history}"),
            ("human", "{question}")
        ])
        chain = prompt_template | llm | StrOutputParser()
        response = chain.invoke({"chat_history": chat_history, "question": question})
        return {"generation": response, "sources": []} # Ensure sources is empty

    def route_question(state: GraphState):
        """Routes the question to either the RAG pipeline or a conversational response."""
        print("---Executing Node: route_question---")
        question = state["question"]
        prompt_template = ChatPromptTemplate.from_template("""Given the user's question below, classify it as either "rag" or "conversational".
        Do not respond with more than one word.
        - "rag": For questions that require specific information from a knowledge base (like the Constitution of Pakistan).
        - "conversational": For greetings, thank yous, or other conversational filler.
        Question: {question}
        Classification:""")
        router_chain = prompt_template | llm | StrOutputParser()
        result = router_chain.invoke({"question": question})
        print(f"Routing classification: {result}")
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

    return workflow.compile()

# --- 4. Main Streamlit App Logic ---

# --- Page Configuration ---
st.set_page_config(page_title="Chat with the Constitution of Pakistan", page_icon="🇵🇰", layout="centered")

# --- Sidebar ---
with st.sidebar:
    st.title("🇵🇰 Chat with the Constitution")
    st.markdown("""
    This app is a chatbot that can answer questions about the Constitution of the Islamic Republic of Pakistan.
    It uses a Retrieval-Augmented Generation (RAG) pipeline to provide accurate answers based on the document's content.
    """)
    if st.button("Clear Chat History", use_container_width=True, type="primary"):
        st.session_state.messages = []
        st.rerun()
    st.markdown("---")
    st.markdown("**Source:** [Constitution of Pakistan (PDF)](https://www.pakp.gov.pk/wp-content/uploads/2024/07/Constitution.pdf)")


# --- Main Content ---
st.title("Chat with the Constitution of Pakistan 🇵🇰")
st.markdown("I can answer questions about the Constitution of Pakistan. Try asking me something!")

# --- Load the RAG graph ---
graph = load_graph()

# --- Initialize session state for messages ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Display previous messages ---
for message in st.session_state.messages:
    avatar = "🧑‍💻" if message["role"] == "user" else "🇵🇰"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])
        # Display sources if they exist for an assistant message
        if message["role"] == "assistant" and "sources" in message and message["sources"]:
            with st.expander("View Sources"):
                for source in set(message["sources"]):
                    st.info(f"Source: {os.path.basename(source)}")


# --- Handle user input ---
if prompt := st.chat_input("Ask a question..."):
    # Add user message to session state and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)

    # Prepare chat history for the chain
    chat_history_for_chain = []
    for msg in st.session_state.messages[:-1]:
        if msg["role"] == "user":
            chat_history_for_chain.append(HumanMessage(content=msg["content"]))
        else:
            chat_history_for_chain.append(AIMessage(content=msg["content"]))

    # --- Stream the response from the graph ---
    with st.chat_message("assistant", avatar="🇵🇰"):
        final_state = {}
        # Use a status indicator to show the process
        with st.status("Thinking...", expanded=False) as status:
            inputs = {"question": prompt, "chat_history": chat_history_for_chain}
            # The stream method yields the state after each node execution
            for event in graph.stream(inputs):
                if "rewrite_query" in event:
                    status.update(label="🔎 Rewriting query for clarity...", state="running", expanded=True)
                if "retrieve_documents" in event:
                    status.update(label="📚 Retrieving relevant articles...", state="running", expanded=True)
                if "generate_rag_response" in event or "generate_conversational_response" in event:
                    status.update(label="🧠 Generating response...", state="running", expanded=True)
                
                # Keep the latest state
                if event:
                    # The event dictionary has one key, which is the name of the node that just ran
                    node_name = list(event.keys())[0]
                    # We store the entire state returned by that node
                    final_state = event[node_name]

            status.update(label="✅ Response generated!", state="complete", expanded=False)

        # --- Display the final response and sources ---
        response = final_state.get("generation", "Sorry, I encountered an error.")
        sources = final_state.get("sources", [])

        st.markdown(response)

        if sources:
            with st.expander("View Sources"):
                for source in set(sources):
                    st.info(f"Source: {os.path.basename(source)}")

        # Add the complete assistant response to session state
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "sources": sources
        })

