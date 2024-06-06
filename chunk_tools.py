from unstructured.partition.pdf import partition_pdf
from unstructured.partition.html import partition_html
from unstructured.partition.text import partition_text
import pandas as pd
from langchain_core.documents import Document
import os
import pickle

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

def get_chunk_documents_from_path(path,i=None,num_paths=None):
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
