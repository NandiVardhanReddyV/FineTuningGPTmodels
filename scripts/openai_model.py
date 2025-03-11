# import os
# import openai
# from langchain_openai import OpenAI
# from langchain.chains import LLMChain
# from langchain.prompts import PromptTemplate
# from dotenv import load_dotenv

# # Load environment variables from .env file
# load_dotenv()

# # Fetch API key from environment variables
# api_key = os.getenv("OPENAI_API_KEY")


# # Set your OpenAI API key
# openai.api_key = 'your-openai-api-key'

# # Generate text using GPT-4
# response = openai.ChatCompletion.create(
#   model="gpt-4",
#   messages=[
#     {"role": "system", "content": "You are a helpful assistant."},
#     {"role": "user", "content": "Write a poem about AI."}
#   ],
#   max_tokens=100
# )

# print("Generated Text:", response['choices'][0]['message']['content'].strip())



# import os
# import openai
# from dotenv import load_dotenv

# # Load environment variables from .env file
# load_dotenv()

# # Fetch API key from environment variables
# api_key = os.getenv("OPENAI_API_KEY")

# # Check if the API key is set
# if not api_key:
#     raise ValueError("OPENAI_API_KEY environment variable not set!")

# # Set the OpenAI API key
# openai.api_key = api_key

# # Generate text using GPT-4
# try:
#     response = openai.ChatCompletion.create(
#         model="gpt-4",
#         messages=[
#             {"role": "system", "content": "You are a helpful assistant."},
#             {"role": "user", "content": "Write a poem about AI."}
#         ],
#         max_tokens=100
#     )
    
#     # Extract and print the response
#     generated_text = response['choices'][0]['message']['content'].strip()
#     print("Generated Text:", generated_text)

# except openai.OpenAIError as e:  # Updated error handling
#     print(f"An OpenAI API error occurred: {e}")

# except Exception as e:  # Catch other unexpected errors
#     print(f"An unexpected error occurred: {e}")


import requests
import json
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY is not set!")

# Define endpoint and headers
url = "https://api.openai.com/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

# Define the request payload
payload = {
    "model": "gpt-4",  # Use a valid model name
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a poem about AI."}
    ],
    "max_tokens": 100,
    "temperature": 0.7
}

# Make the request
try:
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()  # Raise an error for HTTP 4xx/5xx
    result = response.json()
    print("Generated Text:", result['choices'][0]['message']['content'].strip())

except requests.exceptions.RequestException as e:
    print(f"HTTP Request failed: {e}")
