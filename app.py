"""Render web app for the Agentic RAG Router–Retriever notebook.

Keep app.py and trasformer_research_paper-dataset.pdf together in the GitHub repo root.
Set GROQ_API_KEY and TAVILY_API_KEY as Render environment variables.
Start: uvicorn app:app --host 0.0.0.0 --port $PORT
"""
import os
import json
import re
import time
import uuid
import html
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
from pypdf import PdfReader
from crewai import Agent, Crew, Process, Task, LLM
from crewai.tools import tool

APP_DIR = Path(__file__).resolve().parent
PDF_PATH = APP_DIR / "trasformer_research_paper-dataset.pdf"
if not PDF_PATH.is_file():
    raise FileNotFoundError(f"Missing PDF next to app.py: {PDF_PATH.name}")

LLM_MODEL = os.getenv("CREWAI_LLM_MODEL", "groq/openai/gpt-oss-120b")
from crewai.utilities.types import LLMMessage

class GroqCompatibleLLM(LLM):
    def _format_messages_for_provider(self, messages: list[LLMMessage]) -> list[dict[str, str]]:
        cleaned = [
            {key: value for key, value in message.items() if key != "cache_breakpoint"}
            for message in messages
        ]
        return super()._format_messages_for_provider(cleaned)

_llm = None

def get_llm():
    global _llm
    if _llm is None:
        _llm = GroqCompatibleLLM(model=LLM_MODEL, temperature=0)
    return _llm

# Instantiate search backends lazily: the PDF vector index is created only on first PDF question.
_PDF_BACKEND = None
_WEB_BACKEND = None

def get_pdf_backend():
    global _PDF_BACKEND
    if _PDF_BACKEND is None:
        _PDF_BACKEND = PDFSearchTool(
            pdf=str(PDF_PATH),
            config={
                'embedding_model': {
                    'provider': 'onnx',
                    'config': {},
                },
                'vectordb': {
                    'provider': 'chromadb',
                    'config': {},
                },
            },
        )
    return _PDF_BACKEND

def get_web_backend():
    global _WEB_BACKEND
    if _WEB_BACKEND is None:
        from langchain_tavily import TavilySearch
        _WEB_BACKEND = TavilySearch(max_results=5)
        print('Web backend: TavilySearch')
    return _WEB_BACKEND

def format_web_evidence(raw) -> str:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return raw[:14000]
    if isinstance(raw, dict):
        raw = raw.get('results', [raw])
    if not isinstance(raw, list):
        return str(raw)[:14000]
    lines = []
    for index, item in enumerate(raw[:5], 1):
        if isinstance(item, dict):
            title = item.get('title') or f'Result {index}'
            url = item.get('url') or item.get('link') or 'NO_URL'
            excerpt = item.get('content') or item.get('snippet') or ''
            date = item.get('published_date') or 'not specified'
            lines.append(f'[{index}] Title: {title}\nURL: {url}\nPublished: {date}\nEvidence: {str(excerpt)[:2200]}')
        else:
            lines.append(f'[{index}] {str(item)[:1000]}')
    return '\n\n'.join(lines)

class RouteDecision(BaseModel):
    route: Literal['PDF', 'WEB', 'GENERAL'] = Field(description='Exclusive retrieval destination')
    reason: str = Field(description='Short user-facing classification justification, not private reasoning')

LOG_PATH = Path('/tmp/agentic_rag_trace.jsonl')
TRACE_EVENTS = []

def event(run_id: str, actor: str, action: str, **details):
    record = {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(timespec='milliseconds'),
        'run_id': run_id,
        'actor': actor,
        'action': action,
        **details,
    }
    TRACE_EVENTS.append(record)
    with LOG_PATH.open('a', encoding='utf-8') as file:
        file.write(json.dumps(record, ensure_ascii=False, default=str) + '\n')
    return record

def parse_route(task: Task) -> RouteDecision:
    value = getattr(task.output, 'pydantic', None)
    if value is not None:
        return RouteDecision.model_validate(value.model_dump() if hasattr(value, 'model_dump') else value)
    text = getattr(task.output, 'raw', '') or ''
    match = re.search(r'\{[\s\S]*\}', text)
    if not match:
        raise ValueError(f'Router did not produce a parseable decision: {text[:200]}')
    return RouteDecision.model_validate_json(match.group(0))

# 7. Instrumented CrewAI tools. Agent calls are logged by the tool wrapper itself.
def build_crew_tools(run_id: str):
    @tool('pdf_semantic_search')
    def pdf_semantic_search(query: str) -> str:
        """Semantic search ONLY in uploaded Attention Is All You Need PDF. Use for PDF-grounded questions."""
        event(run_id, 'PDFSearchTool', 'tool_started', query=query)
        started = time.monotonic()
        try:
            raw = get_pdf_backend().run(query=query)
            result = str(raw).strip()
            if not result:
                raise RuntimeError('PDF retrieval returned no evidence.')
            event(run_id, 'PDFSearchTool', 'tool_completed', elapsed_sec=round(time.monotonic()-started, 3), evidence_preview=result[:600])
            return f'SOURCE: Attention Is All You Need (uploaded PDF)\nRETRIEVED EXCERPTS:\n{result[:14000]}'
        except Exception as exc:
            event(run_id, 'PDFSearchTool', 'tool_failed', error=str(exc)[:250])
            raise

    @tool('tavily_web_search')
    def tavily_web_search(query: str) -> str:
        """Live Tavily search for current, recent, external information. Return URL-linked evidence."""
        event(run_id, 'TavilySearch', 'tool_started', query=query)
        started = time.monotonic()
        try:
            raw = get_web_backend().invoke({'query': query})
            evidence = format_web_evidence(raw)
            if not evidence.strip():
                raise RuntimeError('Web search returned no evidence.')
            event(run_id, 'TavilySearch', 'tool_completed', elapsed_sec=round(time.monotonic()-started, 3), evidence_preview=evidence[:600])
            return evidence[:14000]
        except Exception as exc:
            event(run_id, 'TavilySearch', 'tool_failed', error=str(exc)[:250])
            raise
    return {'PDF': pdf_semantic_search, 'WEB': tavily_web_search}

async def run_agentic_rag(question: str) -> dict:
    question = question.strip()
    if not question:
        raise ValueError('Question must not be empty.')
    run_id = uuid.uuid4().hex[:12]
    event(run_id, 'User', 'question_received', question=question)

    router = Agent(
        role='Router Agent',
        goal='Classify every question as PDF, WEB, or GENERAL without answering it.',
        backstory='You choose the most appropriate evidence source using the uploaded Transformer paper as the PDF corpus.',
        llm=get_llm(), tools=[], verbose=False, allow_delegation=False,
    )
    route_task = Task(
        description=(
            'Classify this exact user question: {question}\n'
            'Select PDF when the answer is requested from the uploaded paper or concerns its architecture, '
            'methods, experiments or claims. Select WEB for latest/current/2026/recent news or evidence '
            'not contained in the paper. Select GENERAL only for simple creative, arithmetic, or conversational '
            'requests not requiring verification. When PDF content and current facts conflict, choose WEB if '
            'recency is explicitly requested. Do NOT perform retrieval or answer the question. '
            'Return route and a one-sentence reason.'
        ),
        expected_output='JSON object containing route (PDF, WEB, or GENERAL) and short reason.',
        agent=router,
        output_pydantic=RouteDecision,
    )
    begin = time.monotonic()
    try:
        await Crew(agents=[router], tasks=[route_task], process=Process.sequential, verbose=False, memory=False).kickoff_async(
            inputs={'question': question}
        )
        decision = parse_route(route_task)
        event(run_id, 'Router Agent', 'route_selected', route=decision.route,
              reason=decision.reason, elapsed_sec=round(time.monotonic()-begin, 3))
    except Exception as exc:
        event(run_id, 'Router Agent', 'routing_failed', error=str(exc)[:250])
        raise

    selected = decision.route
    tool_lookup = build_crew_tools(run_id)
    selected_tools = [tool_lookup[selected]] if selected != 'GENERAL' else []
    retriever = Agent(
        role='Retriever Agent',
        goal='Use the selected evidence source and answer faithfully with citations.',
        backstory='You are a careful research assistant. Never invent evidence, references, dates or URLs.',
        llm=get_llm(), tools=selected_tools, verbose=False, allow_delegation=False,
        max_iter=5,
    )
    instructions = (
        f'User question: {question}\nRouter selected: {selected}.\n'
        + (
            'MANDATORY: Call your ONE available retrieval tool at least once before answering. '
            'Use ONLY retrieved snippets as evidence. If evidence does not establish an answer, '
            'state what remains unknown. Cite [PDF: Attention Is All You Need] for PDF findings; '
            'do not invent page numbers.' if selected == 'PDF' else
            'MANDATORY: Call your ONE available web search tool at least once before answering. '
            'Rely ONLY on retrieved web evidence, cite the exact source URLs in your answer, '
            'and do not claim a date is verified unless the search results support it.' if selected == 'WEB' else
            'No tools are available. Respond directly. Do not pretend to have checked the PDF or internet.'
        )
    )
    retrieval_task = Task(
        description=instructions,
        expected_output='Concise answer using allowed sources and explicit citations or a clear uncertainty statement.',
        agent=retriever,
        tools=selected_tools,
    )
    begin = time.monotonic()
    try:
        crew_response = await Crew(
            agents=[retriever], tasks=[retrieval_task], process=Process.sequential,
            verbose=False, memory=False, cache=False,
        ).kickoff_async()
        answer = getattr(retrieval_task.output, 'raw', None) or str(crew_response)
        calls = [e for e in TRACE_EVENTS if e['run_id'] == run_id and e['action'] == 'tool_completed']
        if selected != 'GENERAL' and not calls:
            raise RuntimeError('Retriever produced output without a successful call to its required retrieval tool.')
        if selected == 'WEB' and 'http' not in answer:
            event(run_id, 'Verifier', 'citation_warning', warning='No web URL found in answer; review before submission.')
        if selected == 'PDF' and 'Attention Is All You Need' not in answer:
            event(run_id, 'Verifier', 'citation_warning', warning='PDF document title missing in answer; review citation.')
        event(run_id, 'Retriever Agent', 'answer_produced', route=selected,
              elapsed_sec=round(time.monotonic()-begin, 3), answer=answer)
        return {'run_id': run_id, 'route': selected, 'route_reason': decision.reason, 'answer': answer}
    except Exception as exc:
        event(run_id, 'Retriever Agent', 'retrieval_failed', route=selected, error=str(exc)[:250])
        raise


app = FastAPI(title="Agentic RAG Router-Retriever")

STYLE = """
<style>
  body { font-family: system-ui, sans-serif; background:#f7f9fc; margin:0; color:#162c42; }
  main { max-width:820px; padding:30px 22px; margin:40px auto; background:white;
         border-radius:16px; box-shadow:0 6px 25px #25374a12; }
  h1 { margin-bottom:4px; font-size:28px; }
  p.subtitle { margin-top:4px; color:#59677a; }
  textarea { display:block; width:100%; box-sizing:border-box; padding:14px;
             border:1px solid #cad4dc; border-radius:9px; font:inherit; min-height:110px; }
  button { background:#175cd3; color:white; padding:12px 25px; border:0; border-radius:9px;
           font:inherit; font-weight:600; cursor:pointer; margin-top:14px; }
  button:hover { background:#0f4db4; }
  section { margin-top:25px; border-top:1px solid #e5eaf0; padding-top:16px; }
  .route { display:inline-block; font-size:13px; font-weight:700; color:#1e4782;
           background:#eaf1ff; border-radius:10px; padding:4px 12px; }
  .answer { font:inherit; line-height:1.65; white-space:pre-wrap; overflow-wrap:anywhere; }
  .error { background:#fff1f1; color:#941d1d; padding:12px; border-radius:8px; }
  small { color:#6c7888; }
</style>"""


def render_page(question="", result=None, error=None):
    question_safe = html.escape(question, quote=True)
    sections = ""
    if error:
        sections = f'<section><div class="error">{html.escape(error)}</div></section>'
    elif result:
        route = html.escape(result["route"])
        reason = html.escape(result["route_reason"])
        answer = html.escape(result["answer"])
        sections = (f'<section><span class="route">Route: {route}</span>'
                    f'<p><b>Router decision:</b> {reason}</p>'
                    f'<h3>Answer</h3><div class="answer">{answer}</div></section>')
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Agentic RAG — Router / Retriever</title>{STYLE}</head>
    <body><main><h1>Agentic RAG — Router / Retriever</h1>
    <p class="subtitle">CrewAI agents • PDF semantic search • Tavily web search • Groq</p>
    <p>Ask about the included <i>Attention Is All You Need</i> paper, or request recent research.</p>
    <form action="/ask" method="post"><label for="question"><b>Your question</b></label>
    <textarea id="question" name="question" required placeholder="Explain scaled dot-product attention from the PDF">{question_safe}</textarea>
    <button type="submit">Ask</button></form>{sections}
    <section><small>For public deployments, keep API keys in Render Environment settings, never GitHub.</small></section>
    </main></body></html>"""


@app.get("/", response_class=HTMLResponse)
async def home():
    return render_page()


@app.get("/health")
async def health():
    return {"status": "ok", "pdf_found": PDF_PATH.is_file()}


@app.post("/ask", response_class=HTMLResponse)
async def ask(request: Request):
    form = await request.form()
    question = str(form.get("question", "")).strip()
    if not question:
        return render_page(error="Please enter a question.")
    missing = [name for name in ("GROQ_API_KEY", "TAVILY_API_KEY") if not os.getenv(name)]
    if missing:
        return render_page(question, error="Set these keys in Render Environment: " + ", ".join(missing))
    try:
        result = await run_agentic_rag(question)
        return render_page(question, result=result)
    except Exception as exc:
        return render_page(question, error=f"{type(exc).__name__}: {str(exc)[:500]}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "10000")))
