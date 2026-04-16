## 1. Hierarchical Retrieval Tools

The agent is equipped with three primary tools designed to interact with the corpus at different granularities:

- **`keyword_search`**:
    
    - **Purpose**: Performs exact lexical matching to locate chunks containing specific terms.
    - **Input Parameters**: A list of keywords and a parameter k for the number of results.
    - **Output**: Returns the top-k chunk IDs along with **abbreviated snippets** consisting only of sentences containing the keywords.
    - **Logic**: It uses a scoring system based on keyword frequency and length (weighting longer keywords more heavily) to rank chunks.
        
- **`semantic_search`**:
    
    - **Purpose**: Finds passages that are conceptually related or have similar meaning to a natural language query.
    - **Input Parameters**: A natural language query string and a `top_k` integer (default: 5, max: 20).
    - **Output**: Returns top-k chunk IDs and snippets of the most relevant sentences within those chunks.
    - **Logic**: It uses dense vector representations (specifically **Qwen3-Embedding-0.6B**) and cosine similarity to match the query to sentence-level embeddings.
        
- **`chunk_read`**:
    
    - **Purpose**: Allows the agent to access the **complete content** (approx. 1,000 tokens) of specific document segments.
    - **Input Parameters**: An array of chunk IDs (e.g., `['0', '24']`).
    - **Output**: The full text of the requested chunks.
    - **Additional Ability**: The agent can autonomously decide to read adjacent chunks (+/- 1) to gather surrounding context if the information seems truncated.

## 2. Implementation Strategies (Prompt-Level Instructions)

The project provides explicit "STRATEGY" and "IMPORTANT" guidelines within the LLM's system prompt to ensure efficient and accurate retrieval:

- **The "Incomplete Information" Rule**: The prompt explicitly warns the agent that search results from `keyword_search` and `semantic_search` are **NOT sufficient** for answering questions because they only show snippets. The agent **MUST** use `chunk_read` to get the full context before formulating a final answer.
- **Promising Chunk Strategy**: The agent is instructed to always read the full content of "promising chunks" identified during the initial search phase.
- **Context Tracker Notification**: To prevent redundant work, a **Context Tracker** records all previously read chunks. If the agent attempts to read a chunk it has already seen, the tool returns a notification: _"This chunk has been read before"_. This encourages the model to explore new parts of the corpus instead of wasting tokens.
## 3. The Agent Reasoning Logic

The agent follows a **ReAct-like loop** (Reasoning and Acting):

1. **Thought Phase**: The model analyzes the current state of information and determines what is missing.
2. **Action Phase**: It selects exactly **one tool** to call based on the granularity needed (e.g., exact entity name = keyword search; general topic = semantic search).
3. **Observation Phase**: It processes the tool's output and updates its reasoning.
4. **Scaling Effort**: If a model encounters a complex multi-hop question, it can extend this loop for more steps (up to 20 steps in some tests) to refine its search strategy.

## Helpful system Prompt:

### 1. Core System Instruction

This part defines the agent's identity and the overarching strategy of "progressive information disclosure."

> **System Prompt:**
> 
> "You are an intelligent research assistant equipped with hierarchical retrieval tools to answer complex questions from a large corpus. Your goal is to gather sufficient evidence through iterative exploration.
> 
> **CRITICAL GUIDELINES:**
> 
> 1. **Iterative Reasoning:** After each tool call, analyze the results. If the information is incomplete, decide which tool will best help you find the missing piece.
> 2. **Snippet vs. Full Text:** Initial search results (`keyword_search` and `semantic_search`) return only **abbreviated snippets**. These are for identification purposes only and are **NOT sufficient** for answering questions.
> 3. **The 'Read' Requirement:** You **MUST** use the `chunk_read` tool to access the full content of any chunk you find promising before you attempt to formulate a final answer. 4. **Avoid Redundancy:** Use the context history to ensure you are not repeatedly reading the same chunks. If a tool tells you a chunk has been read, pivot your search strategy."
---

### 2. Complete Tool Prompt Descriptions

These descriptions guide the model on _when_ to choose each interface. You would pass these as the "description" field in your OpenAI/LangChain tool definitions.

#### **Tool 1: keyword_search**

- **Prompt Instruction:** "Use this tool for **precise entity matching** or when you know the exact wording of a name, date, or term. It searches the entire corpus for exact lexical matches."
    
- **Implementation Logic:**
    
    - **Parameters:** `keywords` (List of strings), `top_k` (Integer).
    - **Returns:** A list of Chunk IDs and sentences containing the keywords.
#### **Tool 2: semantic_search**

- **Prompt Instruction:** "Use this tool for **conceptual or meaning-based matching**, or when the exact wording in the documents is unknown. It finds passages that are semantically related to your query."
    
- **Implementation Logic:**
    
    - **Parameters:** `query` (Natural language string), `top_k` (Integer).
    - **Returns:** Top-ranked Chunk IDs and relevant semantic snippets.
#### **Tool 3: chunk_read**

- **Prompt Instruction:** "**Essential Tool.** Read the complete content of document segments by their IDs. Use this to examine the full context and details that are hidden in search snippets. **STRATEGY:** If information seems truncated, you may also read adjacent chunks (ID ± 1)."
    
- **Implementation Logic:**
    
    - **Parameters:** `chunk_ids` (List of strings).
    - **Returns:** The full ~1,000 token text of the requested chunks.

---

### 3. Strategy Logic (The "Reasoning" Chain)

In your code, the agent should follow this "Logic Flow" embedded in the prompt:

|**Step**|**Agent Logic**|**Tool Selection Criteria**|
|---|---|---|
|**1. Identify**|Locate potential chunks using specific terms.|**Use `keyword_search`**|
|**2. Explore**|Find related concepts or general context.|**Use `semantic_search`**|
|**3. Verify**|Get the complete, detailed text of found IDs.|**Use `chunk_read`**|
|**4. Finalize**|Combine all full-text evidence into an answer.|**End Loop**|

By providing these specific "IMPORTANT" and "STRATEGY" tags in your system prompt, the LLM will naturally prioritize **identifying** chunks through snippets and **verifying** them through the `chunk_read` tool, which is the core architectural requirement of the A-RAG paper.