from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from streamlit_mic_recorder import speech_to_text
from gtts.lang import tts_langs
import streamlit as st
from gtts import gTTS
import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from PyPDF2 import PdfReader
import docx
from langchain.vectorstores import FAISS
# import sentence_transformers

# Initialize conversation history

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
    
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
    
if 'past' not in st.session_state:
    st.session_state['past'] = []
    
if 'entered_prompt' not in st.session_state:
    st.session_state['entered_prompt'] = ""
    
if "conversation" not in st.session_state:
    st.session_state.conversation = None
    
if "processComplete" not in st.session_state:
    st.session_state.processComplete = None
    
if 'audio_files' not in st.session_state:
    st.session_state.audio_files = []
    
# Function to format chat history as text for download

def format_chat_for_download(chat_history):
    formatted_text = ""
    for i, message in enumerate(chat_history):
        if message ["role"] == "user":
            formatted_text += f"User: {i+1}: {message['content']}\n"
        elif message["role"] == "bot":
            formatted_text += f"Bot {i+1}: {message['content']}\n"
    return formatted_text

# Function to display conversation and play audio

def display_conversation_and_audio():
    for i, message in enumerate(st.session_state.conversation_history):
        if isinstance(message, dict):
            if message["role"] == "user":
                st.markdown(user_template.replace("{{MSG}}", message["content"]), unsafe_allow_html= True)
            elif message["role"] == "bot":
                st.markdown(bot_template.replace("{{MSG}}", message["content"]), unsafe_allow_html=True)
                response_audio_file = f"response_audio_{(i//2)+1}.mp3"
                st.audio(response_audio_file)
            
# Apply Custom CSS

css = '''
<div class="avatar">
        <img src="https://i.ibb.co/C8cyn0P/Red-White-Neon-Circle-Instagram-Profile-Picture-2.png" style="max-height: 150px; max-width: 150px; border-radius: 50%; object-fit: cover;margin-left: 16rem">
    </div> 
<style>
    .chat-message {
        padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex;
    }
    .chat-message.user {
        background-color: #e3a9d5;
    }
    .chat-message.bot {
        background-color: #d689cb;
    }
    .chat-message .avatar img {
        max-width: 78px;
        max-height: 78px;
        border-radius: 50%;
        object-fit: cover;
    }
    .chat-message .message {
        width: 80%;
        padding: 0 1.5rem;
        color: #1c0202;
    }
    body {
        background-color: skyblue !important;
    }
    [data-testid="stSidebar"] {
        background-color: lightgray !important;
    }
</style>
'''

st.markdown(css, unsafe_allow_html = True)

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://i.pinimg.com/originals/0c/67/5a/0c675a8e1061478d2b7b21b330093444.gif" style="max-height: 70px; max-width: 50px; border-radius: 50%; object-fit: cover;">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://th.bing.com/th/id/OIP.uDqZFTOXkEWF9PPDHLCntAHaHa?pid=ImgDet&rs=1" style="max-height: 80px; max-width: 50px; border-radius: 50%; object-fit: cover;">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''

# File upload and processing
def get_files_text(uploaded_files):
    text = ""
    for uploaded_file in uploaded_file:
        _, file_extension = os.path.splitext(uploaded_file.name)
        if file_extension == ".pdf":
            text += get_pdf_text(uploaded_file)
        elif file_extension == ".docx":
            text += get_docx_text(uploaded_file)
        else:
            st.error(f"Unsupported file type:  {file_extension}. Only PDF and DOCX are supported.")
    return text   

def get_pdf_text(pdf):
    try:
        reader = PdfReader(pdf)
        return "".join([page.extract_text() for page in reader.pages if page.extract_text() ])
    except Exception as e:
        st.error(f"Error reading PDF file: {e}")
        return ""
    
def get_docx_text(doc_file):
    try:
        doc = docx.Document(doc_file)
        return ' '.join([pare.text for pare in doc.paragraphs if pare.text])
    except Exception as e:
        st.error(f"Error reading DOCX file: {e}")
        return ""
    
def get_text_chunks(text):
        if len(text) < 100:
            st.warning("Document content too short for meaningful Q/A.")
        splitter = CharacterTextSplitter(separator="\n", chunk_size=1500, chunk_overlap=2000)
        return splitter.split_text(text)
    
def get_vectorstore(text_chunks):
    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_store = FAISS.from_texts(text_chunks, embeddings)
        return vector_store
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        return None
    
# Setup conversation chain
def get_conversation_chain(vectorstore, api_key):
    try:
        model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)
        return ConversationalRetrievalChain.from_llm(
            llm = model,
            retriever=vectorstore.as_retriever(),
        )
    except Exception as e:
        st.error(f"Error setting up conversation chain: {e}")
        return None
    
def handle_user_input(user_question):
    response_container = st.container()
    
    response = st.session_state.conversation({
        'question': user_question,
        'chat_history': st.session_state.chat_history
    })
    
    st.session_state.chat_history = response['chat_history']
    
    with response_container:
        for i, messages in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.markdown(user_template.replace("{{MSG}}", messages.content), unsafe_allow_html=True)
            else:
                st.markdown(bot_template.replace("{{MSG}}", messages.content), ussafe_allow_html=True)

api_key = "AIzaSyAuJKkl00ffWx8oWoctiBAz-7rye3AdwHM"  # Use a placeholder

st.title("🎙️RAG Voice Conversation Bot🤖")

# Sidebar for language selection and   Q/A type

language = st.sidebar.selectbox("Select Language", ["Urdu", "English"])
option = st.sidebar.selectbox("Choose an option", ["General Q/A", "Document Q/A"])

if st.sidebar.button("Clear Chat"):
    st.session_state.conversation_history = []
    st.success("Chat history cleared.")
    
if language == "Urdu":
    chat_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful AI assistant. Please always respond to user queries in Urdu."),
            ("human", "{human_input}"),
        ]
    )
    response_lang = "ur"
else:
    chat_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful AI assistant. Please always respond to user queries in English."),
            ("human", "{human_input}"),
        ]
    )
    response_lang = "en"
    
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)
chain = chat_template | model | StrOutputParser()

if option == "General Q/A":
    st.subheader("General Question and Answer")
    
    if language == "Urdu":
        text = speech_to_text(language="ur", use_container_width=True, just_once=True, key="STT_Urdu")
    else:
        text = speech_to_text(language="en", use_container_width=True, just_once=True, key="STT_English")
        
    user_input = text or st.text_input("Ask something:", key="input")
    
    if user_input:
        st.session_state.past.append(user_input)
        st.session_state.entered_prompt = user_input
        response = chain.invoke({"human_input": user_input})
        
        st.session_state.conversation_history.append({"role": "user", "content": user_input})
        st.session_state.conversation_history.append({"role": "bot", "content": response})
        
        tts = gTTS(text=response, lang= response_lang)
        response_audio_file = f"response_audio_{len(st.session_state.audio_files)+1}.mp3"
        tts.save(response_audio_file)
        st.session_state.audio_files.append(response_audio_file)
        
        display_conversation_and_audio()
        
        # Corrected download button functionality
        if st.download_button("Download Chat History", data=format_chat_for_download(st.session_state.conversation_history).encode("utf-8"), file_name="chat_history.txt", mime="text/plain"):
            st.success("Download started!")
            
# Document Q/A Flow

if option == "Document Q/A":
    st.subheader("Upload Documents for Q/A")
    uploaded_files = st.file_uploader("Upload PDF or DOCX files", type=["pdf", "docx"], accept_multiple_files=True)
            
    if uploaded_files:
        text = get_files_text(uploaded_files)
        if text.strip():
            text_chunks = get_text_chunks(text)
            vector_store = get_vectorstore(text_chunks)
            if vector_store:
                st.session_state.conversation = get_conversation_chain(vector_store, api_key)
                st.session_state.processComplete = True
                st.success("Document processed successfully!")
            else:
                st.error("Failed to process document for Q/A.")
        else:
            st.warning("No content found in the uploaded files.")
            
    question = st.text_input("Ask a question about your documents")
    
    if question and st.session_state.conversation:
        handle_user_input(question)
    elif not st.session_state.conversation:
        st.warning("Please upload and process a document first.")
        
    # Download chat history
    if st.download_button("Download Chat History", data=format_chat_for_download(st.session_state.conversation_history).encode("utf-8"), file_name="chat_history.txt", mime="text/plain"):
        st.success("Download started!")
                