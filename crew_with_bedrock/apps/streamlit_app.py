import streamlit as st
from bedrock_agent_module import run_agent_with_query

# Set page title and configuration
st.set_page_config(page_title="AI Agent with Amazon Bedrock", layout="wide")
st.title("AI Agent with Amazon Bedrock")

# Initialize session state for storing conversation history
if "conversation" not in st.session_state:
    st.session_state.conversation = []

# Initialize query input state if not exists
if "query_input" not in st.session_state:
    st.session_state.query_input = ""

# Function to handle form submission
def handle_submit():
    if st.session_state.query_input:
        query = st.session_state.query_input
        
        # Get response from AI agent
        with st.spinner("AI Agent is thinking..."):
            response = run_agent_with_query(query)
        
        # Store the conversation
        st.session_state.conversation.append((query, response))
        
        # Clear the input box (this is safe because we're setting it before the next rerun)
        st.session_state.query_input = ""

# Main area for conversation
st.header("Chat with AI Agent")

# Display conversation history
for i, (query, response) in enumerate(st.session_state.conversation):
    with st.chat_message("user"):
        st.write(query)
    with st.chat_message("assistant"):
        st.write(response)

# Input area with callback
st.text_input("Ask a question:", key="query_input", on_change=handle_submit if st.session_state.query_input else None)
submit_button = st.button("Submit", on_click=handle_submit)