from Chatbot import ChatBot
import streamlit as st

# Initialize ChatBot instance in session state
if "chatbot" not in st.session_state:
    st.session_state.chatbot = ChatBot()

# Initialize session state for chat messages
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I assist you today?"}]

st.set_page_config(page_title="Chatbot Assistant", layout="centered")

st.title("Chatbot Assistant")

# Sidebar for file processing
st.sidebar.header("File Processing")
file_name = st.sidebar.text_input("Enter file name (e.g., 'example.txt'):")
if st.sidebar.button("Process File"):
    try:
        file_path = f"./data/{file_name}"
        st.session_state.chatbot.process_file(file_path, file_name)
        st.sidebar.success("File processed successfully!")
    except Exception as e:
        st.sidebar.error(f"Error: {e}")

# Sidebar for clearing Pinecone index
if st.sidebar.button("Clear Pinecone Index"):
    try:
        st.session_state.chatbot.clear_index(file_name)
        st.sidebar.success("Pinecone index cleared successfully!")
    except Exception as e:
        st.sidebar.error(f"Error: {e}")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User input for chat
if user_input := st.chat_input("Type your message here..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = st.session_state.chatbot.ask_question(user_input)
                st.write(response)
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Error: {e}")