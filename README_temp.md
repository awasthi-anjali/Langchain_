## Tools file:

Using Tools with LangChain (Step-by-Step)
This example shows how to give an LLM access to a Python function (tool) and let the model decide when to call it.

---

1️⃣ Defining a Tool
from langchain.tools import tool

@tool
def get_weather(location: str) -> str:
"""Get the weather at a location"""
return f"It's sunny in {location}"
What this does
• @tool converts a normal Python function into a LangChain Tool
• The docstring tells the LLM what the tool does
• The type hint (location: str) tells the LLM what input it needs
Now the model understands:
👉 There exists a tool named get_weather
👉 It expects a location argument

---

2️⃣ Binding Tools to the Model
model_with_tools = model.bind_tools([get_weather])
What this does
• Attaches the get_weather tool to the LLM
• Allows the model to decide when to use it
Without this step, the model cannot call tools.

---

3️⃣ Model Generates Tool Calls
response = model_with_tools.invoke("What is the weather in India?")
print(response)
What happens
Instead of answering directly, the model returns:
• A tool name
• Tool arguments
Example internally:
{
"name": "get_weather",
"args": {"location": "India"}
}
This is called a tool call.

---

Viewing Tool Calls
for tool_call in response.tool_calls:
print(f"Tool: {tool_call['name']}")
print(f"Args: {tool_call['args']}")
Output
Tool: get_weather
Args: {'location': 'India'}
Meaning
The LLM decided:

- Call get_weather
- Pass location="India"
