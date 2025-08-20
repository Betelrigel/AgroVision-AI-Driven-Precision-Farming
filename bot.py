import streamlit as st
from ollama import Client
from datetime import datetime

class AgroVisionOllamaChatbot:
    def __init__(self):
        self.client = Client(host='http://localhost:11434')  # Default Ollama host
        self.knowledge_prompt = """
        You are an agricultural expert chatbot named AgroVision, developed by xAI, specializing in rice and wheat cultivation in India, with a focus on Hamirpur, Himachal Pradesh when possible. Provide accurate, concise, and helpful responses based on your knowledge. Include Hamirpur-specific details where applicable (e.g., sowing times, climate suitability). Respond in the user's selected language (English or Hindi). Include the current date and time (e.g., 11:18 PM IST on July 10, 2025) at the end of each response. If the query is unrelated to rice or wheat cultivation, politely redirect the user to ask about these topics.
        """
        self.current_time = datetime.now().strftime("%I:%M %p IST on %B %d, %Y")

    def get_response(self, query, language="English"):
        # Construct the full prompt with user query
        full_prompt = f"{self.knowledge_prompt}\n\nUser query: {query}\nResponse:"
        
        # Call Ollama API with Llama 3 model
        response = self.client.chat(
            model="llama3",  # Use the locally downloaded Llama 3 model
            messages=[
                {"role": "system", "content": full_prompt},
                {"role": "user", "content": query}
            ],
            options={"temperature": 0.7, "max_tokens": 300}  # Adjust as needed
        )
        
        return response['message']['content'].strip()

# Streamlit interface for Ollama chatbot
def show_groq_chatbot():
    st.markdown('<div class="header">Hamirpur Farmer Chatbot 🌾</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subheader">Ask about rice or wheat cultivation in Hamirpur, Himachal Pradesh (English or Hindi)</div>',
        unsafe_allow_html=True
    )
    
    chatbot = AgroVisionOllamaChatbot()
    
    language = st.selectbox("Select Language / भाषा चुनें", ["English", "Hindi"])
    query = st.text_input(
        "Enter your query / अपना प्रश्न दर्ज करें",
        placeholder="E.g., When to sow wheat in Hamirpur? / हमीरपुर में गेहूं कब बोना चाहिए?"
    )
    
    if st.button("Submit Query / प्रश्न सबमिट करें"):
        if query.strip():
            with st.spinner("Generating response..."):
                response = chatbot.get_response(query, language)
                st.markdown(f'<div class="response-box">{response}</div>', unsafe_allow_html=True)
        else:
            st.warning("Please enter a query / कृपया एक प्रश्न दर्ज करें")
    
    st.markdown(
        '<div style="text-align: center; margin-top: 30px; color: #757575;">Powered by xAI\'s Grok and Ollama with Llama 3</div>',
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    show_groq_chatbot()