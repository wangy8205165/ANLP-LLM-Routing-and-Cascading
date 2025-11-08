from openai import OpenAI
api_key = "your api token"
client = OpenAI(api_key=api_key)

response = client.responses.create(
    model="gpt-5",
    reasoning={"effort": "low"},
    instructions="Talk like a pirate.",
    input="Are semicolons optional in JavaScript?",
)

print(response.output_text)
