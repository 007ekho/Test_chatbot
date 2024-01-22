from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
                                SystemMessagePromptTemplate,
                                HumanMessagePromptTemplate,
                                ChatPromptTemplate,
                                MessagesPlaceholder
)
from streamlit_chat import message
from auxiliary_functions import *


import langchain
import streamlit as st
# import os
# os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
openai.api_key = st.secrets.OPENAI_API_KEY

st.header('Welcome, My name is Javis')
st.write("")

if 'responses' not in st.session_state:
    st.session_state['responses'] = ['How can I assist you']
if 'requests' not in st.session_state:
    st.session_state['requests']=[]


llm = ChatOpenAI(model_name= "gpt-3.5-turbo",api_key=openai.api_key)

if 'buffer_memory' not in st.session_state:
    st.session_state.buffer_memory = ConversationBufferWindowMemory(k=3,return_messages=True)

system_msg_template = SystemMessagePromptTemplate.from_template(template="""Your name is Javis.Answer the question as truthfully as possible using the provided context, 
and if the answer is not contained within the text below, say 'I am sorry, It's probably out of context'. Answer the questions as precise as possible. You are not a poem or Haiku writer""")
human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")
prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])

conversation = ConversationChain(memory=st.session_state.buffer_memory, prompt=prompt_template, llm=llm, verbose=True)

response_container = st.container()
textcontainer = st.container()
   
# import streamlit as st

# Initialize session_state if not exists
if 'requests' not in st.session_state:
    st.session_state.requests = []

if 'response' not in st.session_state:
    st.session_state.response = []

with textcontainer:
    query = st.text_input("Hello! How can I help you? ", key="input")
    if query:
        with st.spinner("thinking..."):
            conversation_string = get_conversation_string()
            refined_query = query_refiner(conversation_string, query)
            st.subheader("Refined Query:")
            st.write(refined_query)
            context = find_match(refined_query)
            response = conversation.predict(input=f"Context:\n {context} \n\n Query:\n{query}")
        st.session_state.requests.append(query)
        st.session_state.responses.append(response)

with response_container:
    if st.session_state.responses:
        for i in range(len(st.session_state.responses)):
            message(st.session_state.responses[i], key=str(i))
            if i < len(st.session_state.requests):
                message(st.session_state.requests[i], is_user=True, key=str(i) + '_user')
