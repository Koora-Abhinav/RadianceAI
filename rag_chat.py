import os
from dotenv import load_dotenv
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_community.embeddings import OllamaEmbeddings
from operator import itemgetter

from flask import Flask, render_template, request

load_dotenv()
MODEL = "llama2"

model = Ollama(model=MODEL)

parser = StrOutputParser()
chain = model | parser

loader = PyPDFLoader(r"data/solar_data.pdf")
pages = loader.load_and_split()
pages

template = """
Answer the following questions based on the context below and solar energy and renewable energy. If you can't answer the question, reply "I don't know".

Context: {context}
Question: {question}
"""

prompt = PromptTemplate.from_template(template)

embedding = OllamaEmbeddings()

vectorstore = DocArrayInMemorySearch.from_documents(pages, embedding=embedding)

def get_answer(question):
    retriever = vectorstore.as_retriever()
    retriever.invoke("Solar energy")
    chain = (
        {
            "context": itemgetter("question") | retriever, 
            "question": itemgetter("question")}
        | prompt
        | model
        | parser
    )

    ans = chain.invoke({"question": question})
    return ans

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        question = request.form['question']
        answer = get_answer(question)
        return render_template('index.html', answer=answer)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
