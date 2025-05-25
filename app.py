import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

# Streamlit app setup
st.set_page_config(page_title="🤖 Conversational Chatbot with Memory", page_icon="🤖")
st.title("🤖 Conversational Chatbot with Memory")
st.subheader("💬 Chat with memory using OpenAI API")

# Sidebar inputs
with st.sidebar:
    st.markdown("### ⚙️ Model Parameters")
    temperature = st.slider("🌡️ Temperature", 0.0, 1.0, 0.7, 0.05)
    max_tokens = st.slider("🔢 Max Tokens", 100, 400, 200, 10)
    openai_api_key = st.text_input("🔑 OpenAI API Key", type="password")

if not openai_api_key:
    st.warning("⚠️ Please enter your OpenAI API key in the sidebar to use the chatbot.")
    st.stop()

# Initialize memory and chain in session_state
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize the OpenAI Chat model
llm = ChatOpenAI(
    temperature=temperature,
    max_tokens=max_tokens,
    openai_api_key=openai_api_key,
    model_name="gpt-4o-mini"
)

# Create the ConversationChain with memory
conversation = ConversationChain(
    llm=llm,
    memory=st.session_state.memory,
    verbose=False,
)

# Use a form to submit user input
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("🗨️ You:")
    submitted = st.form_submit_button("🚀 Send")

    if submitted and user_input:
        with st.spinner("🤖 AI is typing..."):
            try:
                # Get AI response using the conversation chain (memory included)
                response = conversation.run(user_input)

                # Save conversation history for UI display
                st.session_state.messages.append({"role": "user", "content": user_input})
                st.session_state.messages.append({"role": "ai", "content": response})

            except Exception as e:
                st.error(f"❌ Error: {e}")

# Display conversation history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"**🗨️ You:** {msg['content']}")
    else:
        st.markdown(f"**🤖 AI:** {msg['content']}")
        st.markdown("---")