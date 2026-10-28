# Agentic RAG — Router/Retriever with PDF and Web Search

## Project overview
This notebook implements an Agentic Retrieval-Augmented Generation (RAG) workflow using **CrewAI** and two role-based agents. The PDF corpus is the attached 15-page research paper **Attention Is All You Need** by Vaswani et al. A Router Agent selects a `PDF`, `WEB`, or optional `GENERAL` route. A Retriever Agent is then constructed with *only* the selected search tool, and returns a source-grounded answer.

## Deliverables
- `Agentic_RAG_Router_Retriever.ipynb` — executable notebook, agent definitions, tools, orchestration, demo questions, audit, trace table, and timing visualization.
- `README.md` — architecture, setup, coordination, trade-offs.
- `requirements.txt` — installation dependencies.
- `trasformer_research_paper-dataset.pdf` — supplied source PDF; keep with notebook when executing.

## Architecture

```
User question
     |
CrewAI Router Agent -> RouteDecision(route, short reason)
     |
     +-- PDF ----- PDFSearchTool (semantic PDF index / ChromaDB)
     +-- WEB ----- langchain_tavily.TavilySearch
     +-- GENERAL - no external retrieval (optional)
     |
CrewAI Retriever Agent (ONLY the chosen tool, or no tool)
     |
Grounded answer + source references
     |
JSONL audit events + notebook trace/timing visualization
```

The routing crew and retrieval crew are **separate sequential CrewAI invocations** by design. The first Crew completes a Router task and returns a validated classification; Python chooses the allowed tool, and a second Crew runs the Retriever task. A single fixed sequential Crew with both tools always available would not enforce tool-level conditional routing. This design makes the boundary explicit and auditable.

## Setup (Python 3.11 recommended)

1. Copy the `.ipynb`, `requirements.txt`, and supplied `.pdf` into the **same folder**.
2. In a terminal opened in the project folder, create and activate an isolated environment, then install the dependencies:

   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   python -m ipykernel install --user --name agentic-rag-venv --display-name "Python (Agentic RAG .venv)"
   ```

   If Python 3.11 is unavailable, use `python -m venv .venv` instead. In VS Code/Jupyter, select the **Python (Agentic RAG .venv)** kernel for this notebook and restart the kernel after installing packages. Avoid installing these dependencies into a different/global Python interpreter; the notebook must use the environment where they were installed.
3. Create Groq and Tavily API keys. **Only `GROQ_API_KEY` and `TAVILY_API_KEY` are required.** Groq runs the default `groq/openai/gpt-oss-120b` chat model, PDF embeddings run locally with ChromaDB's ONNX MiniLM model, and Tavily provides web search. The embedding model may download on first use. Set the keys in your environment or enter them securely at the notebook prompts. Do not commit keys or save them in notebook outputs. Example in a local `.env` file (not included in submission):

   ```text
   GROQ_API_KEY=your_groq_key
   TAVILY_API_KEY=your_tavily_key
   ```

4. Run all notebook cells from top to bottom. The notebook accepts the PDF at its own folder path or `/mnt/data/` in this execution environment.
5. The two demo questions exercise both the **PDF** and **WEB** routes; review the displayed answers, `agentic_rag_trace.jsonl`, trace dataframe, timing visualization, and audit table.

**Costs/access:** Running the demo makes real Groq chat and Tavily API calls, which may incur usage charges. PDF embeddings are generated locally; first PDF access may take longer while the local ONNX model is downloaded and the document is indexed. Network and valid credentials are required.

## Tools, vector store, and role responsibilities

| Component | Technology | Purpose |
|---|---|---|
| Router agent | CrewAI `Agent` + `Task`, typed Pydantic output | Interpret question and select PDF / WEB / GENERAL with short reason |
| PDF retrieval | `crewai_tools.PDFSearchTool` | Embed/chunk and semantically retrieve research-paper excerpts |
| Vector database | **ChromaDB**, configured in PDFSearchTool | Store document embeddings and perform similarity retrieval |
| Embeddings | Local ONNX MiniLM via ChromaDB | Map document passages and query to dense vectors without an external embedding API |
| Web retrieval | `langchain_tavily.TavilySearch` | Retrieve fresh external snippets and URLs using the maintained integration |
| Retriever agent | CrewAI `Agent` + `Task` | Call only selected tool and produce cited answer or admit insufficient evidence |
| Reasoning trace | JSONL and pandas/matplotlib | Display observable decisions, tool invocation, timings, answers, and errors |

## Input / output flow

1. User sends a natural-language question.
2. Router Agent classifies based on source needs: uploaded paper -> PDF; recent/external fact -> WEB; simple non-evidence prompt -> GENERAL.
3. The `RouteDecision` Pydantic object is validated before selecting tools.
4. Retrieval crew contains the Retriever Agent plus **only** the selected tool wrapper. Tool wrapper executes PDFSearchTool or Tavily, logs activity and evidence preview, and returns source text.
5. Retriever writes concise, evidence-grounded answer, citing the uploaded paper or Tavily source URLs. If evidence is inadequate, it is instructed to say so.
6. Logs and audit verify the correct tool was called; a retrieval route that never successfully invoked its tool is treated as a failure.

## Demonstration queries

- PDF: `According to the uploaded Attention Is All You Need paper, explain scaled dot-product attention.` The source equation is in Section 3.2.1, page 4. The source paper's Transformer big model lists 28.4 EN-DE BLEU in Table 2, page 8; this is another suitable PDF query.
- WEB: `What are notable developments in transformer research in 2026? Provide links to recent sources.`
- Optional GENERAL: `Write a two-line motivational quote for students.`

**Do not represent route labels or answers above as measured runtime results**: the notebook generates the actual results only after live API execution. No model keys were supplied when this submission template was authored.

## Challenges, trade-offs, and limitations

- **Correct conditional routing vs simplicity:** Two CrewAI crew runs add overhead, but guarantee the Retriever only receives an authorized tool rather than relying on prompt-only tool selection.
- **Semantic PDF search:** Chroma similarity retrieval is practical and low-configuration, but complex equations or page numbering can be lost in chunk extraction. The answer is cited at document level unless verified page metadata is present.
- **Fresh web results:** Tavily gives source URLs, but snippets can be incomplete; answer prompts prohibit uncited invention. Search result publish dates are not always available.
- **Tavily integration:** The notebook uses the maintained `langchain_tavily.TavilySearch` package rather than the deprecated `langchain-community` class.
- **Tool observability vs reasoning privacy:** The trace stores router classification reasons and observable tool interactions; it intentionally does **not** reveal private model reasoning. Logs can contain user queries and returned answer text—retain securely.
- **Reliability:** A successful tool invocation does not certify each factual claim; add automatic claim-level evidence checks, hybrid search, reranking, page-aware indexing, and human review for production.

## How to review the notebook submission

A successful run should show (a) PDF routed to PDFSearchTool, (b) fresh query routed to Tavily, (c) source references in results, (d) router/retriever tool call logs including timing, and (e) audit checks passing. If the LLM routes unexpectedly, inspect the short routing reason and adjust its classification instructions; routing is probabilistic.

## References

- CrewAI documentation: https://docs.crewai.com/
- PDFSearchTool description/configuration: https://crewai.ubitools.com/tools/file-document/pdfsearchtool/
- LangChain-Tavily reference: https://python.langchain.com/docs/integrations/tools/tavily_search/
- Tavily LangChain migration guidance: https://github.com/langchain-ai/langchain-skills/blob/main/config/skills/langchain-dependencies/SKILL.md
- Paper: Vaswani et al., *Attention Is All You Need*, https://arxiv.org/abs/1706.03762
