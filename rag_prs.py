# Local Imports
from extract_json import extract_json_codeblock
from extract_message_and_json import extract_message_and_json 

# Standard Library Imports
import os
import glob
import uuid
import pickle

# Third Party Imports
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.html import partition_html
from unstructured.partition.text import partition_text
import pandas as pd
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.callbacks import get_openai_callback
# from langchain_community.vectorstores.elasticsearch import ElasticsearchStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
  RunnablePassthrough
)
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
import tiktoken

summary_model = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-1106")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
rag_model = ChatOpenAI(temperature=0, model="gpt-4-turbo")

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

def num_tokens_str(s,model='gpt-3.5-turbo'):
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(s))

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


def smart_partition_file(filename):
    pickle_path = f'{filename}.pkl'
    raw_data = ''
    if os.path.exists(pickle_path):
        print(f'Loading {filename} from cache: {pickle_path}')
        # Load data from the existing pickle file
        with open(pickle_path, 'rb') as f:
            # raw_htmls_elements.append(pickle.load(f))
            raw_data =  pickle.load(f)
    elif filename.endswith('.html'):
        print(f'partitioning html: {filename}...')
        raw_data =  partition_html(filename)
    elif filename.endswith('.pdf'):
        print(f'partitioning pdf: {filename}...')
        raw_data =  partition_pdf(filename=filename,
                                  extract_images_in_pdf=False,
                                  infer_table_structure=True,
                                  chunking_strategy="by_title",
                                  max_characters=1800,
                                  new_after_n_chars=1500,
                                  combine_text_under_n_chars=1000,
                                  )
    elif filename.endswith('.txt'):
        print(f'partitioning text: {filename}...')
        raw_data =  partition_text(filename)
    with open(pickle_path, 'wb') as f:
        pickle.dump(raw_data, f)
    return raw_data

def get_chunks(raw_data):
    chunks = []
    for text_chunk in raw_data:
        if "unstructured.documents.elements.Table" in str(text_chunk.__class__):
            type = 'table'
            text = str(text_chunk.metadata.text_as_html)
        else:
            # print('using text')
            type = 'text'
            text = str(text_chunk)
        chunks.append({
            'type': type,
            'text': text,
            'metadata': text_chunk.metadata,
            'raw_data': text_chunk
        })
    return chunks

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


def doc_data_from_path(path,i=None,num_paths=None):
    if i is not None and num_paths is not None:
        print(f'chunking {i+1}/{num_paths}...')
    filename = os.path.basename(path)
    company = os.path.dirname(path).split('/')[-1]
    document_type = os.path.dirname(path).split('/')[-2]
    document_chunks = get_chunks(smart_partition_file(path))
    document = {
        'document_type': document_type,
        'filename': filename,
        'company': company,
        'path': path,
        'chunks': document_chunks
    }
    return document


if len(document_paths) == 0: # no new chunks to read
    chunks_df = old_chunks_df
else:
    print(f'will chunk {document_paths}. Continue?')
    num_paths = len(document_paths)
    document_dicts = [doc_data_from_path(path, i=i, num_paths=num_paths) for i, path in enumerate(document_paths)]

    print('done chunking.')
    print('creating chunks dataframe...')
    chunks_df = pd.DataFrame(document_dicts)
    chunks_df = chunks_df.explode('chunks').reset_index()
    chunks_df = chunks_df.join(chunks_df['chunks'].apply(pd.Series)).drop(['chunks','index'],axis=1)
    print('done.')

    print('removing duplicates...')
    chunks_df = chunks_df.drop_duplicates(subset=['path','text'])
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
    summarize_output = text_summarize_chain.batch(summarize_target, {"max_concurrency": 20})
    print('done.')
    # chunks_df = chunks_df.drop('summary',axis=1)
    chunks_df['summary'] = pd.Series(summarize_output)
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
total_cost = 15.58 + 2.22
chunks_grouped['cost'] = total_cost * chunks_grouped['text_pct']
chunks_grouped['cost_per_document'] = chunks_grouped['cost'] / chunks_grouped['distinct_paths']



documents = summary_docs_from_chunks(chunks_df)

print(f'creating vector store...')

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

top_files = files_vectorstore.similarity_search_with_relevance_scores("neptune", k=10)
print('show me... NEPTUNE!')
print(top_files[0])
# summary_docs = vectorstore.similarity_search("What is Pfizer's partnership with 23 and me?", k=15)
# top_chunks = vectorstore.similarity_search("What was Roivant's gross-to-net on VTAMA?",k=5)

# TODO chain to get filter 

log={}
def get_context_from_question(user_question,k=30):
    top_files = files_vectorstore.similarity_search(user_question, k=3)
    global_top_files = top_files
    print('top files:',top_files)
    # filter = {"$and": [{"company": "roivant"}, {"$or": [{"document_type": "annual-finaneial-report"}, {"document_type": "quarterly-financial-report"}]}]}
    print('searching vector db...')
    log['top_files'] = top_files
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

# user_question = "What is Pfizer's collaboration with 23 and me?"
# context_summary_docs = get_context_from_question(user_question)

def map_summaries_to_docs(context_summary_docs):
    context_ids = [{'id': doc.metadata['id']} for doc in context_summary_docs]
    context_chunks_df = pd.DataFrame(context_ids).merge(chunks_df,on='id')
    context_docs = docs_from_chunks(context_chunks_df)
    return context_docs

# context_ids = [{'id': doc.metadata['id']} for doc in context_summary_docs]
# context_chunks_df = pd.DataFrame(context_ids).merge(chunks_df,on='id')
# context_docs = docs_from_chunks(context_chunks_df)
context_chain = RunnablePassthrough() | get_context_from_question | map_summaries_to_docs | parse_context_from_docs

# context_chain.invoke(user_question)

rag_prompt_text = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question in as many words as required.
Numbers in tables from financial filings should be assumed to be in thousands unless specified otherwise in the table. 
The current date is 2024-06-03. 
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

# rag_model = ChatOpenAI(temperature=0, model="gpt-4-turbo")
rag_chain = (
  {"question": RunnablePassthrough(), "context": context_chain}
  | rag_prompt
  # | conversation
  | rag_model
  | StrOutputParser()
)

question_answer_chain = (
  {"question": RunnablePassthrough(), "context": RunnablePassthrough()}
  | rag_prompt
  # | conversation
  | rag_model
  | StrOutputParser()
)

def do_rag(user_question):
    top_documents = get_context_from_question(user_question)
    context_docs = map_summaries_to_docs(top_documents)
    context_str = parse_context_from_docs(context_docs)
    rag_prompt_str = rag_prompt.messages[0].prompt.template
    print('Tokens used for rag: ', num_tokens_str(rag_prompt_str + context_str + user_question))
    with get_openai_callback() as cb:
        result= question_answer_chain.invoke([user_question,context_str])
        print(cb)
    try:
        message,chunks_json = extract_message_and_json(result)
        # print(f'used chunks: {chunks_json}')
        # for chunk_index in chunks_json['chunk_indices']:
        #     print(log['top_k'][chunk_index])
    except Exception as e:
        print(f'error parsing chunk json')
        chunks_json = {'chunk_indices': []}
        message = result
        pass
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
    # map history, a list of user-message, assistant-message pairs, to a list of dictionaries
    if len(history) > 0:
        conversation_history = [{'user_message': user, 'assistant_response': assistant} for user, assistant in history]
        extract_question_from_history_prompt_text = f"""You are an assistant tasked with distilling a user's question. You will be given a history of user questions and assistant responses in chronological order. Your task is to use the user's most recent question along with the conversation history to extract the user's question along with all relevant context into one message. For example, if the user asked "What is AI?" followed by "and how is it used?" , you would return the distilled message "How is AI used?" . If the user's question does not relate to the conversation history, just return the user's most recent question as-is (do not involve other questions)
    User Question: {question}
    Conversation History: {conversation_history}
    """
        question = rag_model.invoke(extract_question_from_history_prompt_text).content
        print(f'extracted question: {question}')
    coerce_timeframes_prompt_text=f'''
    You are an assistant tasked with converting timeframes in a user's question to the appropriate document type. The current date is 2024-06-03. You will be given a user question that contains a timeframe such as "last year" or "last quarter". Replace any references to 'last year' with 'using the most recent annual report (10-K)', and replace any references to 'last quarter' with 'using the most recent quarterly report (10-Q).' 
    If no timeframe is present, leave the question as-is. DO NOT modify the user's question unless they reference a timeframe like 'last year' or 'last quarter' or something equivalent.
    User's question: {question}
    '''
    print(f'pre timeframes question: {question}')
    question = rag_model.invoke(coerce_timeframes_prompt_text).content
    print(f'post timeframes question: {question}')
    result, used_chunks = do_rag(question)
    chunk_ids = [chunk.metadata['id'] for chunk in used_chunks]
    # get chunks from chunk_df using chunk_ids
    used_chunks_df = chunks_df[chunks_df['id'].isin(chunk_ids)]
    used_chunks_paths = used_chunks_df['path'].unique().tolist()
    # convert relative paths in used_chunks_paths to absolute paths
    # path_base = 'file:///home/paul/eastwood'
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

result = answer_question("What were the last reported sales for Pfizer's COVID vaccine?")
