import os
import streamlit as st
from constants import google_api_key
from langchain_google_genai import ChatGoogleGenerativeAI

os.environ["GOOGLE_API_KEY"] = google_api_key

st.title("LangChain Demo with Gemini API")

input_text = st.text_input("Search the topic you want")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=0.8
)

if input_text:
    response = llm.invoke(input_text)
    st.write(response.content)