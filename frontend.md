# Role
You are an expert Senior UI/UX Designer and Frontend Architect. You have access to Stitch MCP tools. Your objective is to design and generate a complete, production-ready frontend for a RAG (Retrieval-Augmented Generation) application named **"Ask Data"**. 

# The Objective
Create the screens for the "Ask Data" platform. The application requires a classic dark theme and is split into two primary areas: a Main User Application and an Admin Panel. You will use the Stitch MCP tools to build these screens systematically.

# Execution Strategy (The Stitch Loop)
Please follow this exact workflow to build the application:
1. **Create Project:** Initialize the "Ask Data" project.
2. **Setup Design System:** Create a global design system for a **Classic Dark Theme** (deep grays/blacks for backgrounds, high-contrast white/light gray typography, and a distinct primary accent color like a muted electric blue or emerald green).
3. **Generate Screens:** Generate the screens one by one using `generate_screen_from_text`.
4. **Refine & Polish:** Refine the screens if necessary using variants or edit tools.

# Core Requirements & Features

## 1. Main User Application (The Chat Interface)
- **Login Screen:** Basic fixed ID/Password login (no sign-up required).
- **RAG Chat Interface (Main Workspace):**
  - **Header:** Shows the name "Ask Data", User Profile, and an indicator showing the *Currently Active LLM Model*.
  - **Left Sidebar (Database Context):** A toggleable list showing all available/uploaded files in the database.
  - **Main Chat Area:** Standard conversational UI.
  - **Mentioning System:** Above or within the chat input block, include a "@ Mention" button/feature that allows users to select and scope their queries to specific PDFs instead of the whole database.
  - **Agent Reasoning UI ("Thinking" Block):** When a user submits a query, show a prominent loading/reasoning state (similar to Gemini/ChatGPT). It should visualize the backend tasks (e.g., "Keyword Search...", "Reading Chunks...", "Synthesizing...").
  - *Added Value - Citations:* Ensure chat responses include UI elements for source citations (e.g., small clickable reference badges like [1], [2]).
  - *Added Value - Chat History:* A collapsible drawer or sidebar for past chat sessions.

## 2. Admin Panel (The Management Dashboard)
- **Admin Login:** Fixed ID/Password gate.
- **Admin Dashboard & Ingestion Interface:**
  - **Header & Navigation:** Simple tab navigation (Data Ingestion, Documents, Settings).
  - **Data Ingestion Area:** A drag-and-drop zone to upload new PDFs. 
  - *Added Value - Upload States:* Show status indicators for ingestion (e.g., "Parsing Document", "Extracting Tables", "Generating Embeddings"). Include a validation message system for duplicate uploads (e.g., "Error: Document already exists").
  - **Document Library (List View):** A data table displaying all ingested PDFs, their upload timestamps, token sizes, and status.
  - **Settings/Configuration:** A dedicated screen or modal where the Admin can:
    - Select the active Embedding Model and Generation Model (LLM) from a dropdown.
    - View and securely update API keys.
  - *Added Value - Metrics Widget:* A small overview showing Total Documents, Total Queries, and API Token Usage.

# Instructions for Claude
Your first task is to acknowledge these requirements. Then, immediately execute Step 1 and Step 2 of the Execution Strategy. Once the design system is successfully created, proceed to generate the "User Login" and "Main Chat Interface" screens and present them.
