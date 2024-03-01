import json
from langchain import HuggingFacePipeline
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders import PDFLoader
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter

import transformers
from transformers import AutoTokenizer
import torch


def lambda_handler(event, context):
    # TODO implement
    # Define LLM
    model = "meta-llama/Llama-2-7b-chat-hf"
    
    tokenizer = AutoTokenizer.from_pretrained(model)
    
    pipeline = transformers.pipeline(
        "text-generation", #task
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        max_length=1000,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id
    )
    
    llm = HuggingFacePipeline(pipeline = pipeline, model_kwargs = {'temperature':0})
    
    email_content = {"body_text" : event['EmailBodyText'], 
                    "subject" : event['EmailSubject'], 
                    "sender" : event['EmailSenderName']}
    
    
    # Define prompt
    prompt_template = """ {context} 
    As a customer representative, I need to write an friendly and professional email response to a 
    customer email I have received. 
    This is the email from the customer: """ + email_content["body_text"] 
    + "This is the email subject from the customer: " 
    + email_content["subject"] 
    + "This is the email sender name of the customer: " 
    + email_content["sender"] 
    + """\n\n Most emails can be categorized into 4 types: Product, Orders, General, and Request for quotation
        Please write a professional reponse email based on the customer email."""
    
    prompt = PromptTemplate(template=prompt_template, input_variables=["context"])
    prompt.format(
        context = """My company, Nemat International ® is a manufacturer and distributor of Indian Attars, 
                Perfume oils, Natural perfumes, essential oils, aromatherapy products and more."""
    )
    
    # for doc in docs:
    #     doc.metadata = " "
    
    docs = []
    
    loader = CSVLoader('31 Products Fragrances Descriptions & Notes.csv')
    docs = loader.load()
    
    loader = PDFLoader('Nemat About us.pdf')
    docs += loader.load()
    
    
    # split it into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(docs)
    
    # create the open-source embedding function
    embedding_function = OpenAIEmbeddings()
    #embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # load it into Chroma
    db = Chroma.from_documents(docs, embedding_function)
    # response = stuff_chain.run(docs)
    
    chain_type_kwargs = {"prompt": prompt}
    
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs=chain_type_kwargs,
    )
    
    query = "Please write a response email to the customer given the context, prompt, and relevant data"
    response = chain.run(query)

    return {
        'statusCode': 200,
        'body': json.dumps(response)
    }
    
    






