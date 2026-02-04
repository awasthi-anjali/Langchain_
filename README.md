## Chatbot docs:

### Conversational AI with LangChain, Groq, and Message History

Overview
This project demonstrates how to build a stateful conversational AI using LangChain and Groq LLMs.
It covers prompt design, message history handling, multi-session memory, language control, and context trimming for long conversations.
The implementation shows how an AI can remember past interactions, handle multiple users/sessions, and respond consistently across turns.

---

Tech Stack
• Python
• LangChain
• Groq (LLM provider)
• dotenv (environment variable management)

---

Environment Setup
Environment variables are loaded using python-dotenv to securely manage the Groq API key.
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
This avoids hardcoding sensitive credentials in code.

---

Language Model (LLM)
ChatGroq is used as the chat-based LLM.
model = ChatGroq(
model="llama-3.1-8b-instant",
groq_api_key=groq_api_key
)
This model supports conversational inputs using structured chat messages.

---

Chat Messages
LangChain message objects are used to represent conversations:
• HumanMessage – user input
• AIMessage – assistant response
• SystemMessage – system-level instructions
This allows precise control over conversation flow and context.

---

Basic Invocation
The model can be invoked directly using a list of messages.
model.invoke([HumanMessage(content="Hi, My name is Anjali")])
This is stateless and does not retain memory.

---

Message History (Memory)
To enable memory, RunnableWithMessageHistory is used.
Session Store
store = {}

def get_session_history(session_id: str):
if session_id not in store:
store[session_id] = ChatMessageHistory()
return store[session_id]
Each session ID maintains its own conversation history.

---

Stateful Conversations
with_message_history = RunnableWithMessageHistory(
model,
get_session_history
)
Messages are automatically stored and retrieved per session.
Changing the session_id creates a new independent conversation.

---

Prompt Engineering
ChatPromptTemplate is used to define structured prompts.
prompt = ChatPromptTemplate.from_messages(
[
("system", "You are a helpful assistant."),
MessagesPlaceholder(variable_name="messages")
]
)
This separates system behavior from user messages.

---

Chains
A LangChain pipeline is created using:
chain = prompt | model
This allows prompts and models to be composed cleanly.

---

Language Control
The prompt supports dynamic language selection.
("system", "Answer all questions in {language}")
Language is passed at runtime:
{"messages": [...], "language": "Hindi"}

---

RunnableWithMessageHistory + Chain
When using a chain with message placeholders, the input key must be specified.
RunnableWithMessageHistory(
chain,
get_session_history,
input_messages_key="messages"
)
This ensures LangChain knows where to read chat messages from.

---

Context Trimming
To control token usage and avoid long histories, trim_messages is used.
trimmer = trim_messages(
max_tokens=30,
strategy="last",
token_counter=model,
include_system=True,
start_on="human"
)
Only the most relevant recent messages are kept.

---

RunnablePassthrough
RunnablePassthrough dynamically modifies inputs before passing them to the chain.
RunnablePassthrough.assign(
messages=itemgetter("messages") | trimmer
)
This enables real-time message trimming without changing the original prompt.

---

Multi-Session Memory
Different session IDs maintain independent memory.
config = {"configurable": {"session_id": "chat6"}}
The same user can continue a conversation while another session starts fresh.

---

Key Concepts Demonstrated
• Stateless vs stateful LLM calls
• Chat message abstraction
• Session-based memory
• Prompt templates with placeholders
• Language-controlled responses
• Context window optimization
• Runnable pipelines in LangChain

---

Use Cases
• Conversational assistants
• Multi-user chat systems
• Agentic AI workflows
• Interview-ready LangChain projects
• Memory-aware chatbots

## VECTOR RETRIEVER

VectorStores, Retrievers, and RAG

VectorStores
from langchain_chroma import Chroma

vectorstore = Chroma.from_documents(
documents,
embedding=embeddings
)
What it is
A VectorStore stores documents as numerical embeddings so that semantic (meaning-based) search can be performed.
Why it is used
• Converts text into vectors using embeddings
• Enables similarity-based retrieval instead of keyword search
• Forms the foundation of Retrieval-Augmented Generation (RAG)

---

Similarity Search
vectorstore.similarity_search("cat")
What it does
• Finds the most semantically similar documents to the query "cat"
Why it matters
• Allows the system to retrieve relevant context even if exact words don’t match
• Improves answer quality compared to plain LLM prompting

---

Async Similarity Search
await vectorstore.asimilarity_search("cat")
What it does
• Performs similarity search asynchronously
Why it is useful
• Improves performance in web apps or APIs
• Prevents blocking during I/O-heavy operations
• Essential for scalable, production-grade systems

---

Similarity Search with Scores
vectorstore.similarity_search_with_score("cats")
What it does
• Returns documents along with similarity scores
Why it is useful
• Helps evaluate relevance
• Useful for filtering, ranking, or debugging retrieval quality

---

Creating a Runnable Retriever Manually
from langchain_core.runnables import RunnableLambda

retriever = RunnableLambda(
vectorstore.similarity_search
).bind(k=1)

retriever.batch(["cat", "dog"])
What this does
• Wraps similarity_search into a Runnable
• Fixes k=1 using .bind()
• Enables batch execution
Why this is important
• VectorStores are not Runnable
• LCEL (LangChain Expression Language) requires Runnables
• This allows retrieval to be composed into chains and pipelines

---

Using a Built-in Retriever (Recommended)
retriever = vectorstore.as_retriever(
search_type="similarity",
search_kwargs={"k": 1}
)

retriever.batch(["cat", "dog"])
What it is
• A standard LangChain Retriever abstraction
Why it is preferred
• Cleaner and more maintainable
• Implements invoke, batch, ainvoke, abatch
• Fully compatible with LCEL chains

---

RAG (Retrieval-Augmented Generation)
Prompt Definition
from langchain_core.prompts import ChatPromptTemplate

message = """
Answer the question using the provided context only.

{question}
Context:
{context}
"""

prompt = ChatPromptTemplate.from_messages([
("human", message)
])
Why
• Ensures the LLM answers only using retrieved documents
• Prevents hallucinations
• Enforces grounded responses

---

RunnablePassthrough
from langchain_core.runnables import RunnablePassthrough
What it does
• Passes the original user question unchanged into the chain
Why it is needed
• Allows multiple inputs (question, context) in LCEL
• Keeps the question available at inference time

---

RAG Chain Composition
rag_chain = {
"context": retriever,
"question": RunnablePassthrough()
} | prompt | llm
How this works

1. User question enters the chain
2. Retriever fetches relevant context
3. Prompt formats question + context
4. LLM generates an answer grounded in retrieved data

---

Invoking the RAG Chain
response = rag_chain.invoke("tell me about dogs")
print(response.content)
Why this is powerful
• Combines retrieval + reasoning
• Produces accurate, context-aware answers
• Core pattern behind production RAG systems

---

Key Concepts Demonstrated
• VectorStores vs Retrievers
• Sync and async retrieval
• Runnable abstraction
• Batch processing
• LCEL pipelines
• Retrieval-Augmented Generation (RAG)

---

Why This Matters
This implementation shows how raw vector stores can be converted into runnable retrievers and composed into LCEL pipelines, enabling scalable, context-aware RAG systems with clean separation between retrieval, prompting, and generation.
