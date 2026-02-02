import os
from dotenv import load_dotenv
from langchain_community.llms import Ollama
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

#Langsmith Tracking
os.environ["LANGSMITH_API_KEY"]=os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_TRACING"]="true"
os.environ["LANGSMITH_PROJECT"]=os.getenv("LANGSMITH_PROJECT")

#prompt Template
prompt =ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistance.Please respond to the question asked"),
        ("human","Question:{Question}")
    ]
)

#streamlit Framework
st.title("Langchain Demo With LLAMA2")
input_text = st.text_input("What question you have in mind?")

#OLLama Llama2 model
llm=Ollama(model="gemma:2b")
output_parser=StrOutputParser()
chain=prompt|llm|output_parser

if input_text:
    st.write(chain.invoke({"Question":input_text}))