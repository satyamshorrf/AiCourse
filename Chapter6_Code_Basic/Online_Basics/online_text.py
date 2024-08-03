
from openai import OpenAI
from apikey import apikey
import os

os.environ['OPENAI_API_KEY'] = apikey
OpenAI.api_key = apikey

client = OpenAI()
prompt = "Which city is the largest in the world?"

response =client.chat.completions.create(
      model="gpt-3.5-turbo-instruct1",
      messages=[{"role": "user", "content": prompt}])
print("Response")
print(response)
print("Message Content:")
print(response.choices[0].message.content)
