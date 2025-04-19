import streamlit as st
import scripts

# Set page configuration
st.set_page_config(page_title="Bot Assistant", layout="wide")

# Night mode toggle
mode = st.toggle("Dark mode")

# Apply custom CSS for dark or light theme
if mode:
    st.markdown("""
        <style>
        body {
            background-color: #0e1117;
            color: white;
        }
        .stApp {
            background-color: #0e1117;
            color: white;
        }
        .chat-bubble {
            background-color: #1e293b;
            color: white;
            padding: 10px;
            border-radius: 10px;
            margin-bottom: 10px;
            max-width: 70%;
        }
        .chat-bubble.user {
            background-color: #2563eb;
            color: white;
            margin-left: auto;
        }
        </style>
        """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
        body {
            background-color: white;
            color: black;
        }
        .stApp {
            background-color: white;
            color: black;
        }
        .chat-bubble {
            background-color: #f1f5f9;
            color: black;
            padding: 10px;
            border-radius: 10px;
            margin-bottom: 10px;
            max-width: 70%;
        }
        .chat-bubble.user {
            background-color: #3b82f6;
            color: white;
            margin-left: auto;
        }
        </style>
        """, unsafe_allow_html=True)

# Header
st.title("💬 Bot Assistant")

# Load PDF files from the context folder
context_folder = "context"
pdf_files = scripts.load_pdfs_from_context_folder(context_folder)
if pdf_files:
    embeddings, all_texts = scripts.create_embeddings_from_files(pdf_files, file_type='pdf')
    index = scripts.build_faiss_index(embeddings)
else:
    st.error(f"No PDF files found in the '{context_folder}' folder.")

# Chat interface
if pdf_files:
    st.write("### Chat with the Assistant")

    # Initialize session state for chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display the chat history
    chat_placeholder = st.container()
    with chat_placeholder:
        for chat in st.session_state.chat_history:
            if chat["role"] == "user":
                st.markdown(f'<div class="chat-bubble user"><b>You:</b> {chat["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-bubble"><b>Assistant:</b> {chat["content"]}</div>', unsafe_allow_html=True)

    # Input for user query at the bottom
    user_query = st.text_input("Type your question here...", placeholder="Ask me anything...")

    if user_query:
        # Generate a response using the RAG source and the Ollama model
        response = scripts.handle_user_query(user_query, index, all_texts, scripts.getModel())
        
        # Append the question and response to the chat history
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        st.session_state.chat_history.append({"role": "assistant", "content": response})

        # Update the chat history dynamically
        with chat_placeholder:
            st.markdown(f'<div class="chat-bubble user"><b>You:</b> {user_query}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="chat-bubble"><b>Assistant:</b> {response}</div>', unsafe_allow_html=True)
else:
    st.warning("Please add PDF files to the 'context' folder to enable the chat functionality.")