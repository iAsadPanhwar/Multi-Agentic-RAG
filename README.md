
# **AI Question Routing and Retrieval System with Langgraph**

This project demonstrates a powerful AI-driven question-routing and retrieval system, capable of determining the best source of information—either from a vector store (a specialized retrieval source) or Wikipedia (for general knowledge queries). Using LangChain, LangGraph, Cassio, and Groq, this system routes questions intelligently to provide the best possible answers. It implements Retrieval-Augmented Generation (RAG) principles, structured data routing, and AI agent creation, making it an advanced example of AI-driven question answering.

---

## **Project Overview**

This project explores:

1. **Question Routing:** Routing questions to the appropriate source based on their content.
2. **RAG System:** Using vector databases and embeddings to retrieve specialized information.
3. **Wikipedia Integration:** Leveraging Wikipedia for general-purpose questions.
4. **AI Agent Creation:** Using LangGraph to create an autonomous AI system to perform intelligent data retrieval based on input queries.

## **Project Objectives**

1. **Demonstrate Retrieval-Augmented Generation (RAG) with LangChain** - RAG combines traditional search and retrieval techniques with generative models to answer specific questions.
2. **Build an AI Question Router** - Directs user queries to either a vector store with domain-specific documents or to a Wikipedia search for general knowledge.
3. **Integrate Hugging Face for Embeddings** - Uses `sentence-transformers/all-MiniLM-L6-v2` to generate dense vector embeddings for document representation.
4. **Implement AI Agent with LangGraph** - Establish a conditional graph state machine that routes and retrieves relevant answers based on AI decisions.

---

## **Tools and Libraries Used**

* **LangChain** - For creating the question routing and retrieval system.
* **LangGraph** - For graph-based AI agent creation.
* **Cassio** - Cassandra-based vector storage for document embeddings.
* **Hugging Face Embeddings** - For embedding document texts.
* **Wikipedia API** - For general-purpose question answering.
* **Groq** - Used for deploying and querying an LLM model, Llama 3.


## **Project Setup**

### **Prerequisites**

Ensure the following libraries are installed:

```
!pip install -q langchain langgraph cassio langchain_community langchain-groq langchain-huggingface tiktoken wikipedia
```

### **Setup for Vector Database with Cassandra**

Initialize the vector database to store embeddings from the specialized document corpus. You’ll need:

1. **Astra DB Application Token and Database ID** - These authenticate your Cassandra instance for vector storage.

```import
import cassion
ASTRA_DB_APPLICATION_TOKEN = "Your_Astra_DB_Token"
ASTRA_DB_ID = "Your_Astra_DB_ID"

cassio.init(
    token=ASTRA_DB_APPLICATION_TOKEN,
    database_id=ASTRA_DB_ID
)
```

### **Data Preparation and Embedding Generation**

1. **Document Loading** : Load documents from specific URLs.
2. **Text Splitting** : Use LangChain's `RecursiveCharacterTextSplitter` to split documents into manageable chunks for vectorization.
3. **Embedding Creation** : Generate embeddings using Hugging Face’s `sentence-transformers/all-MiniLM-L6-v2`.

```
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

docs = [WebBaseLoader(urls).load() for url in urls]
doc_list = [item for sublist in docs for item in sublist]
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000)
docs_split = text_splitter.split_documents(doc_list)

### **Vector Storage**
```


Store document embeddings in the Cassandra vector store.

```
from langchain.vectorstores import Cassandra
astra_vector_store = Cassandra(
    embedding=embeddings,
    table_name="qa_mini_demo",
    keyspace=None,
    session=None
)

astra_vector_store.add_documents(docs_split)
```

### **Question Routing and Retrieval**

Define the routing logic:

* **Structured LLM Router** - Llama 3 model from Groq determines the routing source for each question (vector store or Wikipedia).
* **Routing Conditions** - Route specific questions to the vector store (specialized) and general questions to Wikipedia (open domain).

```
from langchain_groq import ChatGroq
from google.colab import userdata

os.environ["GROQ_API_KEY"] = userdata.get('GROK_LLAMA3')
llm = ChatGroq(model_name="Llama-3.1-8b-Instant")

system = """
You are an expert at routing a user question to a vectorstore or wikipedia.
The vector store contains documents related to agents, prompt_engineering, and adversial attacks.
Use the vector store for questions on these topics. Otherwise, use wiki-search.
"""

route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

question_router = route_prompt | structured_llm_router
```




### **Create AI Agent with LangGraph**

Define a graph with conditional routing for question handling. The graph routes the question through either vector retrieval or Wikipedia search.

```
from langgraph.graph import StateGraph, START, END

workflow = StateGraph(GraphState)

workflow.add_node("wiki_search", wiki_search)
workflow.add_node("retrieve", retrieve)
workflow.add_conditional_edges(START,
                               route_question,
                               {"wiki_search": "wiki_search",
                                "vectorstore": "retrieve"})
workflow.add_edge("retrieve", END)
workflow.add_edge("wiki_search", END)

app = workflow.compile()
```

---

## Graph


![System Architecture](assets/system-architecture.png)

## **Results**

1. **Routing Results** - Questions on specific topics (e.g., "What is an agent?") were routed to the vector store, fetching relevant, in-depth information from documents.
2. **General Questions** - Broader questions (e.g., "Who is GM Syed?") were routed to Wikipedia, providing concise, encyclopedic summaries.
3. **Efficiency in Information Retrieval** - By combining specialized and open-domain retrieval, the system could handle diverse question types effectively.

---

## **Sample Queries and Outputs**

1. **Query** : "What is an agent?"

* **Route** : Vector Store
* **Response** : Detailed content on agents, including system overview and components.

1. **Query** : "Who is GM Syed?"

* **Route** : Wikipedia
* **Response** : Summary about the Sindhi politician GM Syed.

1. **Query** : "Avengers"

* **Route** : Wikipedia
* **Response** : Summary about the movie  *Avengers: Infinity War* .

---

## **Conclusion**

This system integrates the strengths of RAG, intelligent routing, and open-domain Wikipedia knowledge retrieval to build a robust, multi-faceted question-answering tool. By distinguishing between domain-specific queries and general knowledge questions, it provides accurate and efficient responses, demonstrating advanced techniques in natural language understanding, retrieval, and generative AI.

---

## **Future Improvements**

1. **Broader Document Corpus** : Expand the vector store with additional domain-specific documents.
2. **Fine-Tuning Routing Logic** : Refine the routing system to further improve accuracy.
3. **Enhanced Embedding Models** : Experiment with advanced embedding models for even better retrieval quality.
