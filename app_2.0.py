# app.py

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

# --- 1. Load Environment Variables ---
load_dotenv()

# --- 2. Define the Graph State ---
class GraphState(TypedDict):
    question: str
    generation: str
    chat_history: List[BaseMessage]
    documents: List[str]
    sources: List[str]

# --- 3. Cached Chain Loader ---
@st.cache_resource
def load_chain():
    CHROMA_PATH = "chroma"
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    retriever = db.as_retriever(k=3)
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

    # --- Node Functions ---
    def rewrite_query(state: GraphState):
        question = state["question"]
        chat_history = state["chat_history"]
        if not chat_history:
            return {"question": question}

        rewrite_prompt = ChatPromptTemplate.from_messages([
            ("system", "Rephrase follow-up questions into standalone questions using chat context."),
            ("placeholder", "{chat_history}"),
            ("human", "Rephrase this follow-up question into a standalone one:\n{question}")
        ])
        rewriter = rewrite_prompt | llm | StrOutputParser()
        rewritten_question = rewriter.invoke({"chat_history": chat_history, "question": question})
        return {"question": rewritten_question}

    def retrieve_documents(state: GraphState):
        question = state["question"]
        documents = retriever.invoke(question)
        doc_contents = [doc.page_content for doc in documents]
        sources = [doc.metadata.get("source", "N/A") for doc in documents]
        return {"documents": doc_contents, "sources": sources}

    def generate_response_rag(state: GraphState):
        question = state["question"]
        documents = state["documents"]
        chat_history = state["chat_history"]
        context_text = "\n\n---\n\n".join(documents)

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", 
             "You are a helpful assistant that answers based on the provided Constitution context. "
             "If the context doesn’t have the info, say it's not available in the document.\n\nContext:\n{context}"),
            ("placeholder", "{chat_history}"),
            ("human", "{question}")
        ])

        chain = prompt_template | llm | StrOutputParser()
        response = chain.invoke({"context": context_text, "chat_history": chat_history, "question": question})
        return {"generation": response}

    def generate_conversational_response(state: GraphState):
        question = state["question"]
        chat_history = state["chat_history"]
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are a friendly and conversational chatbot."),
            ("placeholder", "{chat_history}"),
            ("human", "{question}")
        ])
        chain = prompt_template | llm | StrOutputParser()
        response = chain.invoke({"chat_history": chat_history, "question": question})
        return {"generation": response, "sources": []}

    def route_question(state: GraphState):
        question = state["question"]
        prompt_template = ChatPromptTemplate.from_template(
            "Classify the question as 'rag' if it needs document info, or 'conversational' otherwise.\nQuestion: {question}\nClassification:"
        )
        router_chain = prompt_template | llm | StrOutputParser()
        result = router_chain.invoke({"question": question})
        return "conversational" if "conversational" in result.lower() else "rag"

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

# --- 4. Streamlit App UI ---
st.set_page_config(page_title="Chat with the Constitution of Pakistan 🇵🇰", layout="centered")
st.title("📜 Chat with the Constitution of Pakistan 🇵🇰")

st.markdown(
    """
    <div style="color:gray; font-size:15px; margin-bottom:10px;">
        Source: 
        <a href="https://www.pakp.gov.pk/wp-content/uploads/2024/07/Constitution.pdf" target="_blank">
            Constitution of the Islamic Republic of Pakistan (PDF)
        </a>
    </div>
    """,
    unsafe_allow_html=True
)

# --- Load Chain ---
chain = load_chain()

# --- Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Options")
    if st.button("🧹 Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    st.markdown("---")
    st.markdown("**Tip:** Ask questions like:")
    st.markdown("- *What are the fundamental rights in the Constitution?*")
    st.markdown("- *When was the Constitution enacted?*")
    st.markdown("- *What is the role of the President?*")

# --- Display Chat History ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Chat Input ---
if prompt := st.chat_input("Ask a question about the Constitution..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Thinking... 💭"):
        chat_history_for_chain = [
            HumanMessage(content=msg["content"]) if msg["role"] == "user" else AIMessage(content=msg["content"])
            for msg in st.session_state.messages[:-1]
        ]

        inputs = {"question": prompt, "chat_history": chat_history_for_chain}
        final_state = chain.invoke(inputs)
        response = final_state["generation"]

        st.session_state.messages.append({"role": "assistant", "content": response})

        with st.chat_message("assistant"):
            st.markdown(response)
