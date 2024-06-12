# Local Imports
from extract_message_and_json import extract_message_and_json 
from chunk_tools import get_chunk_documents_from_path

# Standard Library Imports
import os
import glob
import uuid

# Third Party Imports
import pandas as pd
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.callbacks import get_openai_callback
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain.chains import LLMChain, SimpleSequentialChain

summary_model = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-1106")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
rag_model = ChatOpenAI(temperature=0, model="gpt-4-turbo")

# TODO: put vectorstore in separate vectorstore python file
vectorstore = Chroma(
    collection_name="roiv_documents_v5",
    embedding_function=embeddings,
    persist_directory="/eastwood/db/documents"
)

files_vectorstore = Chroma(
    collection_name="roiv_files_v5",
    embedding_function=embeddings,
    persist_directory="/eastwood/db/files"
)

# Convert the chunk_df to a list of Document objects
# Each Document object will have the page_content as the summary + text of the chunk
# This helps with vectorization
def summary_docs_from_chunks(chunk_df):
    documents = []
    for i, row in chunk_df.iterrows():
        metadata = row[['document_type','filename','company','id']].to_dict()
        documents.append(Document(page_content=row['summary'] + ': ' + row['text'], metadata=metadata))
    return documents

def summary_docs_from_distinct_files(distinct_files):
    documents = []
    for i, row in distinct_files.iterrows():
        doc_repr = f"Document Type: {row['document_type']}, Filename: {row['filename']}, Company: {row['company']}, File Summary: {row['file_summary']}"
        metadata = row[['document_type','filename','company']].to_dict()
        documents.append(Document(page_content=doc_repr, metadata=metadata))
    return documents

def docs_from_chunks(chunk_df):
    documents = []
    for i, row in chunk_df.iterrows():
        metadata = row[['document_type','filename','company','id']].to_dict()
        documents.append(Document(page_content=row['text'], metadata=metadata))
    return documents

categories = [
    ('press-release', 'txt'),
    ('annual-financial-report', 'html'),
    ('quarterly-financial-report', 'html'),
    ('earnings-call', 'pdf'),
    ('sellside-report', 'pdf')
]

document_paths = [
    path for category, extension in categories
    for path in glob.glob(f'./output/{category}/*/*.{extension}')
]

if os.path.exists('chunks.csv'):
    old_chunks_df = pd.read_csv('chunks.csv')
    # get the document paths that are not in the old_chunks_df
    document_paths = [path for path in document_paths if path not in old_chunks_df['path'].tolist()]
else:
    old_chunks_df = pd.DataFrame()

if len(document_paths) == 0: # no new chunks to read
    chunks_df = old_chunks_df
else:
    print(f'will chunk {document_paths}. Continue?')
    num_paths = len(document_paths)
    document_dicts = [get_chunk_documents_from_path(path, i=i, num_paths=num_paths) for i, path in enumerate(document_paths)]

    print('done chunking.')
    print('creating chunks dataframe...')
    chunks_df = pd.DataFrame(document_dicts)
    chunks_df = chunks_df.explode('chunks').reset_index()
    chunks_df = chunks_df.join(chunks_df['chunks'].apply(pd.Series)).drop(['chunks','index'],axis=1)
    print('done.')

    print('removing duplicates...')
    chunks_df = chunks_df.drop_duplicates(subset=['path','text']).reset_index().drop('index',axis=1)
    print('done.')
    print('chunks to summarize: ')
    print(chunks_df)
    print('Continue?')
    input()
    text_summarize_prompt_text = """You are an assistant tasked with summarizing text. \ 
    You will be given a chunk of a file with metadata about the file on the first line of the text \
    Please summarize the text
    Text chunk: {element} """
    text_summarize_prompt = ChatPromptTemplate.from_template(text_summarize_prompt_text)
    text_summarize_chain = {"element": RunnablePassthrough()} | text_summarize_prompt | summary_model | StrOutputParser()
    summarize_target = list('From ' + chunks_df['company']+ "'s" + ' ' + chunks_df['document_type'] + ': ' + chunks_df['filename'] + '\n' + chunks_df['text'])
    print('summarizing chunks...')
    # Get the summary of each chunk in parallel (20 at a time)
    summarize_output = text_summarize_chain.batch(summarize_target, {"max_concurrency": 20})
    print('done.')
    # chunks_df = chunks_df.drop('summary',axis=1)
    chunks_df['summary'] = pd.Series(summarize_output)
    # exit(1) if there are NA values in the summary column
    if chunks_df['summary'].isna().sum() > 0:
        print('Something went wrong in applying the summaries to the chunks. exiting.')
        print(summarize_output)
        print(chunks_df[chunks_df['summary'].isna()])
        exit(1)
    # set the id column using uuid4
    chunks_df['id'] = [str(uuid.uuid4()) for _ in chunks_df['summary']]

    new_documents = summary_docs_from_chunks(chunks_df)
    print('adding new documents to vector store...')
    vectorstore.add_documents(new_documents)
    print('done.')
    prev_shape = chunks_df.shape[0] + old_chunks_df.shape[0]
    chunks_df = pd.concat([chunks_df,old_chunks_df],ignore_index=True)
    new_shape = chunks_df.shape[0]
    if prev_shape != new_shape:
        print('Something went wrong in merging the new summaries')
        exit(1)

chunks_df.to_csv('chunks.csv',index=False)
chunks_overview = chunks_df[['document_type','path','text']].copy()
chunks_overview['text_len'] = chunks_overview['text'].apply(len)
# drop the 'text' column from chunks_overview
chunks_overview = chunks_overview.drop('text',axis=1)
# group chunks_overview by path and get the sum of text_len and the # of chunks in each group
chunks_grouped = chunks_overview.groupby('document_type').agg(
    total_text_len=pd.NamedAgg(column='text_len', aggfunc='sum'),
    distinct_paths=pd.NamedAgg(column='path', aggfunc=lambda x: x.nunique()),
    row_count=pd.NamedAgg(column='text_len', aggfunc='count')  # You can use any column for counting rows
).reset_index()
chunks_grouped['text_pct'] = chunks_grouped['total_text_len'] / chunks_grouped['total_text_len'].sum()
# TODO: more robust cost caluclations
total_cost = 15.58 + 2.22
chunks_grouped['cost'] = total_cost * chunks_grouped['text_pct']
chunks_grouped['cost_per_document'] = chunks_grouped['cost'] / chunks_grouped['distinct_paths']

distinct_files = chunks_df[['document_type','filename','company','path']].drop_duplicates().reset_index().drop('index',axis=1)
# load the length of the file content at `path` 
summarize_file_prompt_text = """Given the following filename, document type, and company, give a summary of what the file is and the type of information it is likely to include:
{file_info}
"""
file_summarize_prompt = ChatPromptTemplate.from_template(summarize_file_prompt_text)
file_summarize_chain = {"file_info": RunnablePassthrough()} | file_summarize_prompt | summary_model | StrOutputParser()

if not os.path.exists('file_summaries.csv'):
    file_summarize_target = list('Filename: ' + distinct_files['filename']+ '\n' + 'Document Type: ' +distinct_files['document_type'] + '\n' + 'Company: ' + distinct_files['company'])
    print('summarizing files...')
    file_summarize_output = file_summarize_chain.batch(file_summarize_target, {"max_concurrency": 10})
    print('done.')
    file_summarize_series = pd.Series(file_summarize_output)
    distinct_files['file_summary'] = file_summarize_series
    distinct_files.to_csv('file_summaries.csv',index=False)
else:
    print('loading file summaries...')
    distinct_files = pd.read_csv('file_summaries.csv')
    distinct_files['file_summary'] = distinct_files['file_summary'].fillna('')
    print('done.')
distinct_files_docs = summary_docs_from_distinct_files(distinct_files)


log={}
def get_context_from_question(user_question,k=30):
    print('searching vector db...')
    top_k = vectorstore.similarity_search(user_question, k=k)
    log['top_k'] = top_k
    print('done.')
    return top_k

def parse_context_from_docs(docs):
    out_str = ""
    for i,doc in enumerate(docs):
        # TODO: remove id from metadata, not needed
        metadata_dict = doc.metadata
        metadata_dict['chunk_index'] = i
        out_str += str(metadata_dict) + '\n'
        out_str += doc.page_content + '\n'
        out_str += '--------------------------------------------------\n'
    return out_str

def map_summaries_to_docs(context_summary_docs):
    context_ids = [{'id': doc.metadata['id']} for doc in context_summary_docs]
    context_chunks_df = pd.DataFrame(context_ids).merge(chunks_df,on='id')
    context_docs = docs_from_chunks(context_chunks_df)
    return context_docs


rag_prompt_text = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question in as many words as required.
Numbers in tables from financial filings should be assumed to be in thousands unless specified otherwise in the table. 
The current date is 2024-06-06. 
If the user asks a question about a financial metric from "last year", make sure to use their latest annual filing (10-K) document to answer.
If the user asks a question about a financial metric from "last quarter", make sure to use their latest quarterly filing (10-Q) document to answer.
At the bottom of your answer, add a json block listing the chunk indices you used to answer the question. It should look like:
```json
{{ 
    "chunk_indices": [0, 7, 11]
}}
```
Context: {context}
Question: {question}
"""

rag_prompt = ChatPromptTemplate.from_template(rag_prompt_text)

question_answer_chain = (
  {"question": RunnablePassthrough(), "context": RunnablePassthrough()}
  | rag_prompt
  | rag_model
  | StrOutputParser()
)

extract_question_from_history_prompt_text = """You are an assistant tasked with distilling a user's question. You will be given a history of user questions and assistant responses in chronological order. Your task is to use the user's most recent question along with the conversation history to extract the user's question along with all relevant context into one message. For example, if the user asked "What is AI?" followed by "and how is it used?" , you would return the distilled message "How is AI used?" . If the user's question does not relate to the conversation history, just return the user's most recent question as-is (do not involve other questions)
User Question: {question}
Conversation History: {conversation_history}
"""
extract_question_from_history_prompt = ChatPromptTemplate.from_template(extract_question_from_history_prompt_text) 

coerce_timeframes_prompt_text='''
You are an assistant tasked with converting timeframes in a user's question to the appropriate document type. The current date is 2024-06-03. You will be given a user question that contains a timeframe such as "last year" or "last quarter". Replace any references to 'last year' with 'using the most recent annual report (10-K)', and replace any references to 'last quarter' with 'using the most recent quarterly report (10-Q).' 
If no timeframe is present, leave the question as-is. DO NOT modify the user's question unless they reference a timeframe like 'last year' or 'last quarter' or something equivalent.
User's question: {question}
'''
coerce_timeframes_prompt = ChatPromptTemplate.from_template(coerce_timeframes_prompt_text ) 

separate_questions_prompt_text='''
You are an information retrieval assistant tasked with extracting questions to query a vector database. You will be given a message from a user that may contain multiple related or unrelated questions. 
Your task is to extract each question into a separate message, including any relevant context. Each question should be on its own line, i.e. your output should delimit the distinct questions with newlines. 
As an example, if the user asks "What are the capitals of France and Germany?" you would return 
"What is the capital of France?
What is the capital of Germany?"
If the user's message asks for an analysis of multiple things, return the questions that would be asked to get the data needed for the analysis. For example, if the user asked "Compare the populations of France and Germany" you would return 
"What is the population of France?
What is the population of Germany?"
Here is the user's question: {question}
'''
separate_questions_prompt = ChatPromptTemplate.from_template(separate_questions_prompt_text ) 

question_history_chain = (
    {"question": RunnablePassthrough(), "conversation_history": RunnablePassthrough()}
    | extract_question_from_history_prompt
    | rag_model
    | StrOutputParser()
)

question_extraction_chain = (
    {"question": RunnablePassthrough(), "conversation_history": RunnablePassthrough()}
    | coerce_timeframes_prompt
    | rag_model
    | separate_questions_prompt
    | rag_model
    | StrOutputParser()
)

coerce_timeframes_chain = (
    {"question": RunnablePassthrough()}
    | coerce_timeframes_prompt
    | rag_model
    | StrOutputParser()
)

separate_questions_chain = (
    {"question": RunnablePassthrough()}
    | separate_questions_prompt
    | rag_model
    | StrOutputParser()
)

example_history = [
    ['what is Roivant?', 'Roivant is a commercial-stage biopharmaceutical company focused on improving patient lives by accelerating the development and commercialization of important medicines. It operates by creating nimble subsidiaries, known as "Vants," to develop and commercialize its medicines and technologies. Roivant also incubates discovery-stage companies and health technology startups complementary to its biopharmaceutical business.']
]
example_conversation_history = [{'user_message': user, 'assistant_response': assistant} for user, assistant in example_history]
user_question = "Compare Roivant's SG&A expense as a percent of total operating expenses from last year with BridgeBio's"
user_question = question_history_chain.invoke([user_question,str(example_conversation_history)])
extracted_questions = question_extraction_chain.invoke([user_question,example_conversation_history])


def do_rag_multiquestion(original_question,extracted_questions_list):
    aggregated_context = ''
    for question in extracted_questions_list:
        print(f'question: {question}')
        top_documents = get_context_from_question(question)
        context_docs = map_summaries_to_docs(top_documents)
        context_str = parse_context_from_docs(context_docs)
        aggregated_context += context_str
    with get_openai_callback() as cb: # used to get the number of tokens used
        result= question_answer_chain.invoke([original_question,aggregated_context])
        print(cb)
    chunks_json = {'chunk_indices': []}
    message = result
    try:
        message,chunks_json = extract_message_and_json(result)
    except Exception as e:
        print(f'error parsing chunk json')
    try:
        chunk_indices = chunks_json['chunk_indices']
        used_chunks = [context_docs[i] for i in chunk_indices]
    except Exception as e:
        print(f'error extracting used chunks: {e}')
        chunks_json = {'chunk_indices': []}
        used_chunks = []
    print(f'returning: {message},{used_chunks}')
    return message, used_chunks

result,used_chunks = do_rag_multiquestion(user_question,extracted_questions.split('\n'))
print(result)

def do_rag(user_question):
    # context_chain = RunnablePassthrough() | get_context_from_question | map_summaries_to_docs | parse_context_from_docs
    top_documents = get_context_from_question(user_question)
    context_docs = map_summaries_to_docs(top_documents)
    context_str = parse_context_from_docs(context_docs)
    with get_openai_callback() as cb: # used to get the number of tokens used
        result= question_answer_chain.invoke([user_question,context_str])
        print(cb)
    chunks_json = {'chunk_indices': []}
    message = result
    try:
        message,chunks_json = extract_message_and_json(result)
    except Exception as e:
        print(f'error parsing chunk json')

    try:
        chunk_indices = chunks_json['chunk_indices']
        used_chunks = [context_docs[i] for i in chunk_indices]
    except Exception as e:
        print(f'error extracting used chunks: {e}')
        chunks_json = {'chunk_indices': []}
        used_chunks = []
    print(f'returning: {message},{used_chunks}')
    return message, used_chunks



def answer_question(question,history=[]):
    print(f'answering question: {question}')
    print(f'history: {history}')
    if len(history) > 0:
        # map history, a list of user-message, assistant-message pairs, to a list of dictionaries
        conversation_history = [{'user_message': user, 'assistant_response': assistant} for user, assistant in history]
        question = question_history_chain.invoke([question,str(conversation_history)])
        print(f'extracted question: {question}')
    questions = separate_questions_chain.invoke(question)
    print(f'separated questions: {questions}')
    # TODO coerce
    result, used_chunks = do_rag(question)
    chunk_ids = [chunk.metadata['id'] for chunk in used_chunks]
    used_chunks_df = chunks_df[chunks_df['id'].isin(chunk_ids)]
    used_chunks_paths = used_chunks_df['path'].unique().tolist()
    used_chunks_paths = [path.replace('output','../static') for path in used_chunks_paths]
    # get the basename of each chunk path
    used_chunk_links = [f"[{os.path.basename(path)}]({path.replace(' ','%20')})" for path in used_chunks_paths]
    appendix = '\n\nUsed files:\n' + '\n'.join(used_chunk_links)
    result += appendix
    return result

example_history = [
  ['Where is Washington DC?',"I don't know"],
  ['what is Roivant?', 'Roivant is a commercial-stage biopharmaceutical company focused on improving patient lives by accelerating the development and commercialization of important medicines. It operates by creating nimble subsidiaries, known as "Vants," to develop and commercialize its medicines and technologies. Roivant also incubates discovery-stage companies and health technology startups complementary to its biopharmaceutical business.']
]

if __name__ == '__main__':
    user_question = "Compare their SG&A expense from last year with Pfizer's"
    # user_question = "and what are their main drugs?"
    print(f'answering question: {user_question}')
    result = answer_question(user_question,history=example_history)
    print(result)
