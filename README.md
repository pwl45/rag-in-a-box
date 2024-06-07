The file edgarclient.py can be used to scrape financial reports

see line 117 for where it reads documents:
```python3
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
```

once you have some documents, do:

`docker compose up`

`docker exec -it eastwood /bin/bash`

`uvicorn chatbot_interface_fastapi:app --port 7860 --host 0.0.0.0`
