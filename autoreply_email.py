from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os

# Import your existing code here
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.chains.llm import LLMChain
from langchain.chains import RetrievalQA

# Set the OpenAI API key
os.environ["OPENAI_API_KEY"] = "INSERT-OPENAI-API-KEY"

class MyHTTPRequestHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        post_data = json.loads(post_data.decode('utf-8'))

        # Extract sender name, email text, and email subject from the POST data
        sender_name = post_data.get('sender_name')
        email_text = post_data.get('email_text')
        email_subject = post_data.get('email_subject')

        # Your existing code to process the email content and generate a response
        # Define prompt
        prompt_template = """ {context} As a customer representative, I need to write an friendly and professional
        email response to a customer email I have received. This is the email from the customer: """ + email_text + "This is the email subject from the customer: " + email_subject + "\nThis is the email sender name of the customer: " + sender_name + """\n\n Most emails can be categorized into 4 types: Product, Orders, General, and Request for quotation
            Please write a professional reponse email based on the customer email."""

        prompt = PromptTemplate(template=prompt_template, input_variables=["context"])
        prompt.format(
            context = """My company, Fragrances International ® is a manufacturer and distributor of
                    Perfume oils, Natural perfumes, essential oils, aromatherapy products and more."""
        )

        llm = ChatOpenAI(temperature=0)
        
        # Load the documents
        docs = []

        loader = CSVLoader('s3_files/ProductsFragrancesDescriptionsNotes.csv')
        docs += loader.load()

        loader = PyPDFLoader('s3_files/AboutUs.pdf')
        docs += loader.load()

        # Split data into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_documents(docs)

        # Create the open-source embedding function
        embedding_function = OpenAIEmbeddings(model="text-embedding-3-small")

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

        # Send the response back to the client
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        response_data = json.dumps({'response': response})
        self.wfile.write(response_data.encode('utf-8'))

def run_server(server_class=HTTPServer, handler_class=MyHTTPRequestHandler, port=55555):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print(f"Server running on port {port}")
    httpd.serve_forever()

if __name__ == '__main__':
    run_server()

