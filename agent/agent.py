import os
import json

from dotenv import load_dotenv
from openai import OpenAI

from tools.data_tools import agent_instance


load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


tools = [
    {
        "type": "function",
        "name": "get_dataset_shape",
        "description": "Returns the number of rows and columns in the dataset.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }
]


input_messages = [
    {
        "role": "user",
        "content": "How many rows and columns are in the dataset?"
    }
]


response = client.responses.create(
    model="gpt-5-mini",
    input=input_messages,
    tools=tools,
    tool_choice="auto"
)


# Save the LLM response, including the function call
input_messages += response.output


for item in response.output:
    if item.type == "function_call":

        if item.name == "get_dataset_shape":

            result = agent_instance.get_dataset_shape()

            print("Tool result:")
            print(result)

            input_messages.append(
                {
                    "type": "function_call_output",
                    "call_id": item.call_id,
                    "output": json.dumps(result)
                }
            )


final_response = client.responses.create(
    model="gpt-5-mini",
    input=input_messages,
    tools=tools
)

print(final_response.output_text)