from transformers import AutoTokenizer, AutoModelForSeq2SeqLM,pipeline
from langchain.prompts import PromptTemplate
from langchain import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
import streamlit as st



def load_llm(task="text2text-generation",model="./flan-t5-large",temperature=0,max_length=512):
    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModelForSeq2SeqLM.from_pretrained(model)
    pipe = pipeline(task, model=model, tokenizer=tokenizer)
    llm = HuggingFacePipeline(
        pipeline = pipe,
        model_kwargs={"temperature": temperature, "max_length": max_length},
    )
    return llm


def defined_template():
    template = """Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Keep the answer as concise as possible. 
    {context}
    Question: {question}
    Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
    return QA_CHAIN_PROMPT
    
#
def ask_and_get(db,llm,question,k=3):
    QA_CHAIN_PROMPT = defined_template()
    retriever = db.as_retriever(search_type='similarity', search_kwargs={'k': k})
    qa_chain  = RetrievalQA.from_chain_type(   
    llm=llm,   
    chain_type="stuff",   
    retriever=retriever,   
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT} 
    ) 
    result = qa_chain.run({ "query" : question })
    print("9"*21,result)
    return result 
    

# clear the chat history from streamlit session state
def clear_history():
    if 'history' in st.session_state:
        del st.session_state['history']

