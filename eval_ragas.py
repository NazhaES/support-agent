import json
import uuid

from ragas import EvaluationDataset, RunConfig, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference, LLMContextRecall, ResponseRelevancy

from langchain_groq import ChatGroq

from app.core.config import BASE_DIR, get_settings
from app.rag.vectorstore import get_embeddings
from app.rag.workflow import ask

settings = get_settings()

TESTS = json.loads((BASE_DIR / "data" / "eval_dataset.json").read_text(encoding="utf-8"))

rows = []
for test in TESTS:
    question, reference = test["question"], test["reference"]
    result = ask(question, thread_id=f"eval-{uuid.uuid4()}")
    contexts = [d.page_content for d in result["kb_docs"]] if result["source_used"] == "private_kb" else [result["web_results"]]
    rows.append({"user_input": question, "response": result["answer"], "retrieved_contexts": contexts, "reference": reference})
    print(f"{result['source_used']:>12} | {question}")

scores = evaluate(
    EvaluationDataset.from_list(rows),
    metrics=[Faithfulness(), ResponseRelevancy(), LLMContextPrecisionWithReference(), LLMContextRecall()],
    llm=LangchainLLMWrapper(ChatGroq(model=settings.judge_model, temperature=0, api_key=settings.groq_api_key)),
    embeddings=LangchainEmbeddingsWrapper(get_embeddings()),
    run_config=RunConfig(max_workers=2, timeout=120),
)

print(scores)
# scores.to_pandas().to_csv("ragas_results.csv", index=False)
