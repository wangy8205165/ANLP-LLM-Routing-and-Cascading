from openai import OpenAI
import os
api_key = os.environ["OPENAI_API_KEY"]
client = OpenAI(api_key=api_key)

response = client.responses.create(
    model="gpt-5",
    reasoning={"effort": "low"},
    instructions="Talk like a pirate.",
    input="Are semicolons optional in JavaScript?",
)

print(response.output_text)
