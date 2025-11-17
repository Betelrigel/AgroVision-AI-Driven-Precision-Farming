import streamlit as st
import os
import requests
from datetime import datetime

# If a local `.env` file exists, load it so environment variables like
# GROQ_API_KEY and GROQ_API_URL are available during development.
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    # No dotenv available or failed to load; fall back to environment variables
    pass


class AgroVisionGroqChatbot:
    """Simple Groq-compatible chatbot wrapper.

    This class uses environment variables to configure the API:
    - `GROQ_API_KEY`: Bearer token for auth
    - `GROQ_API_URL`: Full endpoint URL for the chat completion API. If not set,
      the class will attempt to use a sensible default placeholder and will
      raise a helpful error if the request fails.
    - `GROQ_MODEL`: Optional model identifier used for default URL construction.

    Note: Groq/Grok provider APIs vary; set `GROQ_API_URL` to the correct
    endpoint for your account (for example the Groq Cloud or xAI Grok endpoint).
    """

    def __init__(self):
        self.api_key = os.environ.get("GROQ_API_KEY")
        self.model = os.environ.get("GROQ_MODEL", "grok")
        # Default URL pattern — override with GROQ_API_URL if your provider differs
        default_url = f"https://api.groq.ai/v1/models/{self.model}/chat"
        self.api_url = os.environ.get("GROQ_API_URL", default_url)

        self.knowledge_prompt = (
            "You are an agricultural expert chatbot named AgroVision, developed by xAI, "
            "specializing in rice and wheat cultivation in India, with a focus on Hamirpur, "
            "Himachal Pradesh when possible. Provide accurate, concise, and helpful responses "
            "based on your knowledge. Include Hamirpur-specific details where applicable "
            "(e.g., sowing times, climate suitability). Respond in the user's selected language "
            "(English or Hindi). Include the current date and time at the end of each response. "
            "If the query is unrelated to rice or wheat cultivation, politely redirect the user."
        )
        self.current_time = datetime.now().strftime("%I:%M %p IST on %B %d, %Y")

        if not self.api_key:
            raise RuntimeError(
                "GROQ_API_KEY environment variable is not set. "
                "Set it to your Groq API key or set GROQ_API_URL and GROQ_API_KEY to point to your provider."
            )

    def get_response(self, query, language="English"):
        full_prompt = f"{self.knowledge_prompt}\n\nUser query: {query}\nResponse:"

        payload = {
            "messages": [
                {"role": "system", "content": full_prompt},
                {"role": "user", "content": query}
            ],
            "temperature": 0.7,
            "max_tokens": 300,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            resp = requests.post(self.api_url, json=payload, headers=headers, timeout=30)
        except requests.RequestException as e:
            raise RuntimeError(f"Failed to reach Groq API at {self.api_url}: {e}") from e

        if resp.status_code != 200:
            # Give a helpful error message including server response
            raise RuntimeError(
                f"Groq API returned status {resp.status_code}: {resp.text}. "
                "Verify GROQ_API_KEY, GROQ_API_URL and that the model exists."
            )

        data = resp.json()

        # Try multiple common response shapes to extract generated text
        text = None
        if isinstance(data, dict):
            # Common patterns: {'choices': [{'message': {'content': '...'}}]} or {'message': {'content': '...'}}
            choices = data.get("choices")
            if choices and isinstance(choices, list) and len(choices) > 0:
                first = choices[0]
                if isinstance(first, dict):
                    msg = first.get("message") or first.get("text")
                    if isinstance(msg, dict):
                        text = msg.get("content")
                    elif isinstance(msg, str):
                        text = msg
            if not text and isinstance(data.get("message"), dict):
                text = data.get("message", {}).get("content")
            if not text and isinstance(data.get("text"), str):
                text = data.get("text")

        if not text:
            # Fallback: stringify the entire response
            text = str(data)

        return text.strip()

# Streamlit interface for Ollama chatbot
def show_groq_chatbot():
    st.markdown('<div class="header">Hamirpur Farmer Chatbot 🌾 / हमीरपुर किसान चैटबॉट</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subheader">Ask about rice or wheat cultivation in Hamirpur, Himachal Pradesh (English or Hindi) / हमीरपुर, हिमाचल प्रदेश में धान या गेहूं की खेती के बारे में पूछें (अंग्रेजी या हिंदी)</div>',
        unsafe_allow_html=True
    )

    # Instantiate the Groq chatbot. This will raise a helpful error if GROQ_API_KEY is not set.
    try:
        chatbot = AgroVisionGroqChatbot()
    except Exception as e:
        st.error(f"Chatbot not available: {e}")
        return

    language = st.selectbox("Select Language / भाषा चुनें", ["English", "Hindi"])
    query = st.text_input(
        "Enter your query / अपना प्रश्न दर्ज करें",
        placeholder="E.g., When to sow wheat in Hamirpur? / हमीरपुर में गेहूं कब बोना चाहिए?"
    )

    if st.button("Submit Query / प्रश्न सबमिट करें"):
        if query.strip():
            with st.spinner("Generating response... / उत्तर जनरेट किया जा रहा है..."):
                try:
                    response = chatbot.get_response(query, language)
                    st.markdown(f'<div class="response-box">{response}</div>', unsafe_allow_html=True)
                except Exception as err:
                    st.error(f"Failed to generate response: {err}")
        else:
            st.warning("Please enter a query / कृपया एक प्रश्न दर्ज करें")

    st.markdown(
        '<div style="text-align: center; margin-top: 30px; color: #757575;">Powered by xAI\'s Grok and Groq</div>',
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    show_groq_chatbot()