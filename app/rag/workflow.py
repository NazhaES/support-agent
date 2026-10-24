import logging
from typing import Literal
from langchain_core.messages import AIMessage, HumanMessage
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, START, END
from app.core.config import get_settings
from app.core.db import checkpointer
from app.rag.state import AgentState, RouteDecision, EvidenceGrade
from app.rag.vectorstore import search_kb

logger = logging.getLogger(__name__)
settings = get_settings()

_llm = None
_web_search = None

def llm():
    global _llm
    if _llm is None:
        if not settings.groq_api_key:
            raise RuntimeError("GROQ_API_KEY is missing")
        _llm = ChatGroq(
            model=settings.llm_model,
            temperature=0,
            api_key=settings.groq_api_key,
        )
    return _llm

def web_search_tool():
    global _web_search
    if _web_search is None:
        if not settings.tavily_api_key:
            raise RuntimeError("TAVILY_API_KEY is missing")
        _web_search = TavilySearch(
            tavily_api_key=settings.tavily_api_key,
            max_results=5,
            topic="general",
            include_answer=True,
            include_raw_content=False,
        )
    return _web_search

def add_trace(state: AgentState, message: str):
    return [*state.get("trace", []), message]

def contextualize(state: AgentState):
    history = state["messages"][:-1][-settings.history_messages:]
    if not history:
        return {}
    conversation = "\n".join(
        f"{'User' if m.type == 'human' else 'Assistant'}: {str(m.content)[:500]}" for m in history
    )
    standalone = llm().invoke(f"""
Rewrite the latest user message as a standalone question that can be understood without the conversation.
Keep its meaning and do not answer it. If it is already standalone, return it unchanged.
Return only the question.
Conversation:
{conversation}

Latest user message: {state['question']}
""").content.strip()
    if standalone == state["question"]:
        return {}
    return {
        "question": standalone,
        "current_query": standalone,
        "trace": add_trace(state, f"Follow-up rewritten → {standalone}"),
    }

def route_question(state: AgentState):
    router = llm().with_structured_output(RouteDecision, method="json_mode")
    decision = router.invoke(f"""
You route messages for an enterprise IT support assistant.
Use kb for questions about company IT policies, VPN, password reset, MFA, laptop setup,
software access, security, email, devices, troubleshooting, or technology support.
Use direct only for greetings, thanks, or casual chat that needs no company knowledge.
Question: {state['question']}
Return valid JSON like {{"route":"kb"}}.
""")
    return {"source_used": decision.route, "trace": add_trace(state, f"Router → {decision.route.upper()}")}

def route_after_router(state: AgentState) -> Literal["retrieve_kb", "direct_answer"]:
    return "retrieve_kb" if state["source_used"] == "kb" else "direct_answer"

def retrieve_kb(state: AgentState):
    docs = search_kb(state["current_query"])
    return {"kb_docs": docs, "trace": add_trace(state, f"Private KB retrieval → {len(docs)} chunks")}

def grade_kb(state: AgentState):
    grader = llm().with_structured_output(EvidenceGrade, method="json_mode")
    context = "\n\n".join(f"Source: {d.metadata.get('source','unknown')}\n{d.page_content}" for d in state["kb_docs"])
    grade = grader.invoke(f"""
You grade evidence for an enterprise IT support assistant.
Question: {state['question']}
Private company KB evidence:\n{context}
Return good only if the evidence is sufficient to answer confidently and specifically.
Otherwise return weak. JSON: {{"grade":"good"}} or {{"grade":"weak"}}.
""")
    return {"kb_grade": grade.grade, "trace": add_trace(state, f"KB evidence grade → {grade.grade.upper()}")}

def after_kb(state: AgentState) -> Literal["generate_from_kb", "search_web"]:
    return "generate_from_kb" if state["kb_grade"] == "good" else "search_web"

def search_web(state: AgentState):
    result = web_search_tool().invoke({"query": state["current_query"]})
    lines, citations = [], []
    if isinstance(result, dict):
        if result.get("answer"):
            lines.append("Search answer: " + result["answer"])
        for item in result.get("results", []):
            title, url, content = item.get("title", ""), item.get("url", ""), item.get("content", "")
            lines.append(f"Title: {title}\nURL: {url}\nContent: {content}")
            citations.append({"title": title or url, "url": url, "type": "web"})
    else:
        lines.append(str(result))
    return {
        "web_results": "\n\n".join(lines),
        "citations": citations,
        "source_used": "web",
        "trace": add_trace(state, "Web fallback → Tavily search"),
    }

def grade_web(state: AgentState):
    grader = llm().with_structured_output(EvidenceGrade, method="json_mode")
    grade = grader.invoke(f"""
Question: {state['question']}
Web evidence:\n{state['web_results']}
Return good if the evidence is sufficient and directly relevant; otherwise weak.
Return valid JSON like {{"grade":"good"}}.
""")
    return {"web_grade": grade.grade, "trace": add_trace(state, f"Web evidence grade → {grade.grade.upper()}")}

def after_web(state: AgentState) -> Literal["generate_from_web", "rewrite_query", "insufficient"]:
    if state["web_grade"] == "good":
        return "generate_from_web"
    if state["retry_count"] < settings.max_retries:
        return "rewrite_query"
    return "insufficient"

def rewrite_query(state: AgentState):
    rewritten = llm().invoke(f"""
Rewrite this IT support question for better private knowledge retrieval and vendor web search.
Preserve intent, add useful technical keywords, do not answer, return only the query.
Question: {state['question']}
""").content.strip()
    return {
        "current_query": rewritten,
        "retry_count": state["retry_count"] + 1,
        "trace": add_trace(state, f"Query rewrite → {rewritten}"),
    }

def generate_from_kb(state: AgentState):
    context = "\n\n".join(f"[Source: {d.metadata.get('source','unknown')}]\n{d.page_content}" for d in state["kb_docs"])
    answer = llm().invoke(f"""
You are an enterprise IT support copilot. Answer ONLY from the private company KB below.
Be concise, actionable, and safe. If steps are present, present them clearly.
Only state facts, steps, and policies that appear in the KB. Do not add general advice,
assumptions, or extra details that are not in the KB. Do not mention where the answer comes from.
Question: {state['question']}\n\nPrivate KB:\n{context}
""").content
    citations = []
    seen = set()
    for d in state["kb_docs"]:
        title = str(d.metadata.get("source", "Private KB")).split("/")[-1]
        page = d.metadata.get("page")
        if page not in (None, ""):
            title = f"{title}, page {int(page)}"
        if title not in seen:
            seen.add(title)
            citations.append({"title": title, "url": "", "type": "private_kb"})
    return {"answer": answer, "messages": [AIMessage(content=answer)], "source_used": "private_kb", "citations": citations, "trace": add_trace(state, "Answer generation → PRIVATE KB")}

def generate_from_web(state: AgentState):
    answer = llm().invoke(f"""
You are an enterprise IT support copilot. The private company KB was insufficient.
Answer ONLY from the web evidence below. Clearly say this is external web information and may need IT validation before changing company-managed systems.
Question: {state['question']}\n\nWeb evidence:\n{state['web_results']}
""").content
    return {"answer": answer, "messages": [AIMessage(content=answer)], "source_used": "web_search", "trace": add_trace(state, "Answer generation → WEB SEARCH")}

def direct_answer(state: AgentState):
    answer = llm().invoke(f"Respond briefly and naturally to: {state['question']}").content
    return {"answer": answer, "messages": [AIMessage(content=answer)], "source_used": "direct", "trace": add_trace(state, "Direct response → no retrieval")}

def insufficient(state: AgentState):
    answer = "I couldn't find enough reliable evidence in the company knowledge base or external search to answer confidently. Please contact the IT help desk or provide more details."
    return {
        "answer": answer,
        "messages": [AIMessage(content=answer)],
        "source_used": "insufficient_evidence",
        "trace": add_trace(state, "Stopped → insufficient reliable evidence"),
    }

def build_graph():
    graph = StateGraph(AgentState)
    for name, fn in {
        "contextualize": contextualize,
        "route_question": route_question,
        "retrieve_kb": retrieve_kb,
        "grade_kb": grade_kb,
        "search_web": search_web,
        "grade_web": grade_web,
        "rewrite_query": rewrite_query,
        "generate_from_kb": generate_from_kb,
        "generate_from_web": generate_from_web,
        "direct_answer": direct_answer,
        "insufficient": insufficient,
    }.items():
        graph.add_node(name, fn)

    graph.add_edge(START, "contextualize")
    graph.add_edge("contextualize", "route_question")
    graph.add_conditional_edges("route_question", route_after_router, {
        "retrieve_kb": "retrieve_kb", "direct_answer": "direct_answer"
    })
    graph.add_edge("retrieve_kb", "grade_kb")
    graph.add_conditional_edges("grade_kb", after_kb, {
        "generate_from_kb": "generate_from_kb", "search_web": "search_web"
    })
    graph.add_edge("search_web", "grade_web")
    graph.add_conditional_edges("grade_web", after_web, {
        "generate_from_web": "generate_from_web", "rewrite_query": "rewrite_query", "insufficient": "insufficient"
    })
    graph.add_edge("rewrite_query", "retrieve_kb")
    graph.add_edge("generate_from_kb", END)
    graph.add_edge("generate_from_web", END)
    graph.add_edge("direct_answer", END)
    graph.add_edge("insufficient", END)
    return graph.compile(checkpointer=checkpointer)

agent_graph = build_graph()

ANSWER_NODES = {"generate_from_kb", "generate_from_web", "direct_answer"}

def initial_state(question: str) -> AgentState:
    return {
        "messages": [HumanMessage(content=question)],
        "question": question,
        "current_query": question,
        "kb_docs": [],
        "web_results": "",
        "kb_grade": "",
        "web_grade": "",
        "answer": "",
        "source_used": "",
        "retry_count": 0,
        "trace": [],
        "citations": [],
    }

def thread_config(thread_id: str):
    return {
        "configurable": {"thread_id": thread_id},
        "metadata": {"thread_id": thread_id},
        "run_name": "chat_trace",
    }

def ask(question: str, thread_id: str = "default"):
    return agent_graph.invoke(initial_state(question), config=thread_config(thread_id))

def ask_stream(question: str, thread_id: str = "default"):
    state = initial_state(question)
    for mode, chunk in agent_graph.stream(state, config=thread_config(thread_id), stream_mode=["updates", "messages"]):
        if mode == "messages":
            message, metadata = chunk
            if metadata.get("langgraph_node") in ANSWER_NODES and isinstance(message.content, str) and message.content:
                yield {"type": "token", "data": message.content}
            continue
        for update in chunk.values():
            if not update:
                continue
            state.update(update)
            if update.get("trace"):
                yield {"type": "trace", "data": update["trace"][-1]}
    yield {
        "type": "done",
        "answer": state["answer"],
        "source_used": state["source_used"],
        "citations": state["citations"],
        "trace": state["trace"],
        "rewritten_query": state["current_query"],
    }
