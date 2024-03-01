import json
from langchain import HuggingFacePipeline
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.document_loaders import UnstructuredFileLoader
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
import transformers
from transformers import AutoTokenizer
import torch
import os



import langchain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.document_loaders import UnstructuredPowerPointLoader
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain


def lambda_handler(event, context):
    # TODO implement
    os.environ["OPENAI_API_KEY"] = "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"

    # email_content = {"body_text" : input_data['EmailBodyText'],
    #                 "subject" : input_data['EmailSubject'],
    #                 "sender" : input_data['EmailSenderName']}
    
    email_content = {"body_text" : "Hi, who is the founder of this company?",
                    "subject" : "Perfume Options",
                    "sender" : "Alisha"}
                    
    # Define prompt
    prompt_template = """ {context} As a customer representative, I need to write an friendly and professional
    email response to a customer email I have received. This is the email from the customer: """ + email_content["body_text"] + "This is the email subject from the customer: " + email_content["subject"] + "\nThis is the email sender name of the customer: " + email_content["sender"] + """\n\n Most emails can be categorized into 4 types: Product, Orders, General, and Request for quotation
      Please write a professional reponse email based on the customer email."""
    
    prompt = PromptTemplate(template=prompt_template, input_variables=["context"])
    prompt.format(
        context = """My company, Nemat International ® is a manufacturer and distributor of Indian Attars,
                Perfume oils, Natural perfumes, essential oils, aromatherapy products and more."""
    )
    
    llm = ChatOpenAI(temperature=0)
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    chain_type_kwargs = {"prompt": prompt}
    
    # Load the documents
    docs = []
    
    loader = CSVLoader('/content/sample_data/31 Products Fragrances Descriptions & Notes.csv')
    docs = loader.load()
    
    loader = PyPDFLoader('/content/sample_data/Nemat About us.pdf')
    docs += loader.load
    
    # Split data into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents
    
    # Create the open-source embedding function
    embedding_function = OpenAIEmbeddings()
    
    # Load embeddings into Chroma
    db = Chroma.from_documents(docs, embedding_function)
    
    # Build LLM chain
    chain_type_kwargs = {"prompt": prompt}
    
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 1}),
        chain_type_kwargs=chain_type_kwargs,
    )
    
    query = "Please write a response email to the customer given the context, prompt, and relevant data"
    response = chain.invoke(query)
    print(response)
                
                
    return {
        'statusCode': 200,
        'body': json.dumps(response)
    }
    
    






