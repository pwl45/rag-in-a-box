import re
import json

def extract_message_and_json(text):
    # Regular expression to find the JSON block and capture any text before it
    pattern = re.compile(r'(.*?)```json\n(.*?)\n```', re.DOTALL)
    match = pattern.search(text)
    
    if match:
        message = match.group(1).strip()  # Text before the JSON block
        json_block = match.group(2).strip()  # JSON text itself
        
        try:
            json_data = json.loads(json_block)  # Parse the JSON text
            return (message, json_data)  # Return both the preceding message and the JSON data
        except json.JSONError:
            print('Invalid JSON')
            return (message, None)
    
    print('No JSON codeblock found')
    return None, None

def test(text):
    return extract_message_and_json(text)
