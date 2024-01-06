import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain import HuggingFacePipeline, PromptTemplate, LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.chains.question_answering import load_qa_chain
import joblib

# Set CUDA_VISIBLE_DEVICES
os.system("export CUDA_VISIBLE_DEVICES=2")

# Initialize the Hugging Face pipeline for text generation
model_name = "tiiuae/falcon-7b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
generation_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=200,
    do_sample=False,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id
)
llm = HuggingFacePipeline(pipeline=generation_pipeline, model_kwargs={'temperature': 0})

# Load PDF and set up vector search
pdf_filepath = "https://niphm.gov.in/IPMPackages/Wheat.pdf"
pdf_loader = PyPDFLoader(pdf_filepath)
documents = pdf_loader.load_and_split()
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
documents = text_splitter.split_documents(documents)
embeddings = HuggingFaceEmbeddings(model_name="intfloat/e5-large-v2")
doc_search = FAISS.from_documents(documents, embeddings)

# Save the setup for later use
setup_data = {
    'llm': llm,
    'doc_search': doc_search,
    'tokenizer': tokenizer,
}

joblib.dump(setup_data, 'model_setup.joblib')