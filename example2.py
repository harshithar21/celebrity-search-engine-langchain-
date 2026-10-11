#just trying to use SequentialChain to create a simple chain of prompts and responses instead of SimpleSequentialChain
import os
import streamlit as st
from constants import google_api_key
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain

# Set Google API key
os.environ["GOOGLE_API_KEY"] = google_api_key

# Streamlit title and input
st.title("Celebrity Search Results")
input_text = st.text_input("Search the topic you want")

# Initialize Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=0.8
)

# First prompt: Get information about the celebrity
first_input_prompt = PromptTemplate(
    input_variables=["name"],
    template="Tell me about celebrity {name}"
)

chain = LLMChain(
    llm=llm,
    prompt=first_input_prompt,
    verbose=True,
    output_key="person"
)

# Second prompt: Find the celebrity's date of birth
second_input_prompt = PromptTemplate(
    input_variables=["person"],
    template="When was the celebrity born? Use this information:\n{person}"
)

chain2 = LLMChain(
    llm=llm,
    prompt=second_input_prompt,
    verbose=True,
    output_key="dob"
)

third_input_prompt = PromptTemplate(
    input_variables=["dob"],
    template="Mention 5 major events happend around\n{dob} in the world"
)
chain3 = LLMChain(
    llm=llm,
    prompt=third_input_prompt,
    verbose=True,
    output_key="events"
)
# Combine both chains sequentially
parent_chain = SequentialChain(
    chains=[chain, chain2, chain3],
    input_variables=['name'],
    output_variables=['person','dob','events'],
    verbose=True
)

# Run the chain and display the result
if input_text:
    response = parent_chain({'name': input_text})

    print("Final Answer:", response)
    st.write(response)
