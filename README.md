# AgenticRAG : Building Autonomous Retrieval-Augmented Generation Pipelines

This repository contains a step-by-step implementation of **Agentic RAG** systems using Python and Jupyter Notebooks. The project demonstrates the evolution from a traditional fixed retrieval pipeline to an autonomous, graph-based agent workflow capable of dynamic decision-making, tool execution, query evaluation, and rewriting.

## Key Features

- **Standard RAG Baseline**: Fixed retrieve-and-generate implementation.
- **Conditional Routing**: Dynamic evaluation to decide *when* retrieval is necessary.
- **Tool-Driven Retrieval**: Integration of custom tools for fetching context from vector stores and external sources.
- **Query Rewriting & Decomposition**: Automatic query evaluation and transformation to improve context accuracy.
- **End-to-End Agentic Workflow**: State-driven orchestration unifying retrieval, reasoning, and generation.

## Repository Structure

```
AgenticRAG/
├── documents/                             # Sample source documents for indexing
├── 1_base_rag.ipynb                       # Traditional RAG pipeline implementation
├── 2_conditinal_retrieval.ipynb           # Adding conditional logic & state routing
├── 3_tool_use_retrieval.ipynb             # Tool usage and integration
├── 4_tool_use_retrieval2.ipynb            # Refinements to tool-based retrieval
├── 5_retrival_eval_and_rewriting.ipynb    # Query evaluation, rewriting & decomposition
├── AgenticRAG.ipynb                       # Complete end-to-end Agentic RAG system
└── README.md                              # Project documentation
```

## Getting Started

### Prerequisites

- Python 3.10+
- Jupyter Notebook / JupyterLab
-  supported LLM provider

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/SyedaNaziaGit/AgenticRAG.git
   cd AgenticRAG
   ```

2. **Install required dependencies**:
   ```bash
   pip install langgraph langchain-openai faiss-cpu python-dotenv
   ```

3. **Set up environment variables**:
   Create a `.env` file in the root directory:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## Execution Sequence

For the best learning experience, follow the notebooks in numerical order:

1. Open `1_base_rag.ipynb` to establish the baseline.
2. Progress through `2_conditinal_retrieval.ipynb` to `5_retrival_eval_and_rewriting.ipynb` to understand individual agentic components.
3. Run `AgenticRAG.ipynb` to test the complete multi-step autonomous workflow.


**Agentic RAG architectures** differ from **basic (traditional) retrieval architectures** primarily through the integration of **autonomous agents** that dynamically control the retrieval and generation workflow.

### Key Differences

* **Pipeline Execution Structure**:
  * **Basic RAG**: Follows a **fixed, linear pipeline**. Every user query automatically triggers a vector database search to fetch context, combines that context with a prompt, and sends it to the LLM to generate an answer.
  * **Agentic RAG**: Replaces static execution with a **dynamic workflow** (often structured as a state graph using frameworks like LangGraph) with conditional branching.

* **Dynamic Reasoning vs. Static Execution**:
  * **Basic RAG**: Cannot evaluate whether a query actually requires external data or if retrieval was successful.
  * **Agentic RAG**: Employs **autonomous agents** that use reasoning capabilities to make real-time decisions about if, where, and how to execute retrieval actions.

* **Core Decision Capabilities**:
  * **When to retrieve**: The agent decides if retrieval is necessary or if the LLM can answer the question directly.
  * **Where to retrieve**: Dynamically routes queries across multiple external data sources, vector databases, or external tools/APIs.
  * **What to retrieve**: Filters and targets the specific information required for the query intent.
  * **How many times to retrieve**: Enables iterative loops, including query evaluation, query rewriting, and query decomposition across multiple steps.

* **Architectural Implementation**:
  * **Basic RAG**: Consists of a simple chunk-and-embed retriever, prompt template, and LLM call.
  * **Agentic RAG**: Requires **state definitions** (tracking question, state flags, and retrieved documents), agent function nodes, and **conditional edge logic** that routes execution dynamically based on query evaluation.
