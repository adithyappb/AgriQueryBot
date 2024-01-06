# cli_interface.py
from langchain import PromptTemplate, LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
import joblib

# Load setup data
setup_data = joblib.load('model_setup.joblib')
llm = setup_data['llm']
doc_search = setup_data['doc_search']
tokenizer = setup_data['tokenizer']

# CLI Interface
print("Agriculture PDF Chat CLI")
print("Type 'exit' to end the chat.")

unique_answers = set()

while True:
    user_question = input("You: ")

    if user_question.lower() == 'exit':
        print("Exiting the chat. Goodbye!")
        break

    # Perform document similarity search
    similar_docs = doc_search.similarity_search(user_question, k=2)

    # Prompting to LLM for QA
    qa_prompt_template = """
    You are a helpful AI assistant. Use the context delimited by triple backticks to answer the question comprehensively.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    If the question is not related to the context, politely respond that you are tuned to only answer questions that are related to the context.

    Context: ```{context}```

    Question: {question}

    Answer:
    """
    qa_prompt = PromptTemplate(template=qa_prompt_template, input_variables=["context", "question"])
    qa_chain = load_qa_chain(llm=llm, chain_type='stuff', prompt=qa_prompt)
    qa_output_dict = qa_chain({"input_documents": similar_docs, "question": user_question}, return_only_outputs=True)
    answer = qa_output_dict['output_text']

    # Check for repetition and print only unique answers
    if answer not in unique_answers:
        unique_answers.add(answer)
        # Display AI's answer
        print('AI:', answer)