from flask import Flask, request, render_template,jsonify
import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

PINECONE_API_KEY = "f6057e5e-ea62-4ce2-a4c1-7890b309000a"

PINECONE_ENV = "gcp-starter"

index_name = 'emotionise-conversations'
pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENV
)

OPENAI_API_KEY = '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'
model_name = 'text-embedding-ada-002'

embed = OpenAIEmbeddings(
    model=model_name,
    openai_api_key=OPENAI_API_KEY
)

text_field = "text"

# switch back to normal index for langchain
index = pinecone.Index(index_name)

vectorstore = Pinecone(
    index, embed.embed_query, text_field
)

llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model_name='gpt-3.5-turbo',
    temperature=0.0
)


app = Flask(__name__)

def get_response(question):
   template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Keep the answer as concise as possible. Reply with greetings for the greetings and ask what can I do for you. 
   {context}
   Question: {question}
   Helpful Answer:"""
   QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],template=template,)

   qa_chain = RetrievalQA.from_chain_type(llm,
                                       retriever=vectorstore.as_retriever(),
                                       return_source_documents=False,
                                       chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
                                       )
   result = qa_chain({"query": question})
   return result["result"]


@app.route('/', methods=['GET','POST'])
def index():
   if request.method == "POST":
      question = request.form['question']
   else:
      question = request.form.get['question']

   response = get_response(question)
   result = {
        "output": response 
    }
   print(request.form)
   
   return jsonify(result=result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)