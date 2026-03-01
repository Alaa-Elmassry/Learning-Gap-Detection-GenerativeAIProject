import os
from dataclasses import dataclass
from typing import Any, Dict

import pandas as pd
import chromadb

from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


@dataclass
class RAGConfig:
    data_path: str = "data/skills_dataset.xlsx"
    chroma_dir: str = "chroma_db"
    collection_name: str = "skills"
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    top_k: int = 8


def _row_to_document(row: pd.Series) -> Document:
    prereqs = []
    for i in range(5):
        col = f"prerequisites/{i}"
        if col in row and pd.notna(row[col]) and str(row[col]).strip():
            prereqs.append(str(row[col]).strip())

    parts = [
        f"skill_name: {row.get('skill_name', '')}",
        f"category: {row.get('category', '')}",
        f"skill_type: {row.get('skill_type', '')}",
        f"difficulty_level: {row.get('difficulty_level', '')}",
        f"learning_time_days: {row.get('learning_time_days', '')}",
        f"job_demand_score: {row.get('job_demand_score', '')}",
        f"salary_impact_percent: {row.get('salary_impact_percent', '')}",
        f"market_trend: {row.get('market_trend', '')}",
        f"future_relevance_score: {row.get('future_relevance_score', '')}",
        f"prerequisites: {', '.join(prereqs) if prereqs else 'None'}",
    ]
    text = "\n".join([p for p in parts if p.split(': ', 1)[-1].strip()])

    metadata = {
    "skill_name": str(row.get("skill_name", "")),
    "category": str(row.get("category", "")),
    "difficulty_level": float(row.get("difficulty_level")) if str(row.get("difficulty_level", "")).strip() else None,
    "learning_time_days": float(row.get("learning_time_days")) if str(row.get("learning_time_days", "")).strip() else None,
    "prerequisites": " | ".join(prereqs) if prereqs else "",
    }
    return Document(text=text, metadata=metadata)


def build_or_load_index(cfg: RAGConfig) -> VectorStoreIndex:
    os.makedirs(cfg.chroma_dir, exist_ok=True)

    embed_model = HuggingFaceEmbedding(model_name=cfg.embed_model)

    client = chromadb.PersistentClient(path=cfg.chroma_dir)
    collection = client.get_or_create_collection(cfg.collection_name)

    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    try:
        existing_count = collection.count()
    except Exception:
        existing_count = 0

    if existing_count and existing_count > 0:
        return VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)

    df = pd.read_excel(cfg.data_path)
    docs = [_row_to_document(df.iloc[i]) for i in range(len(df))]

    index = VectorStoreIndex.from_documents(
        docs,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True,
    )
    return index


def retrieve_context(index: VectorStoreIndex, topic: str, top_k: int = 8) -> Dict[str, Any]:
    retriever = index.as_retriever(similarity_top_k=top_k)
    nodes = retriever.retrieve(topic)

    top_skills = []
    prereq_set = set()
    diffs = []
    times = []

    for n in nodes:
        md = n.node.metadata or {}
        skill = md.get("skill_name") or ""
        if skill:
            top_skills.append(skill)

        pr_str = md.get("prerequisites") or ""
        if isinstance(pr_str, str) and pr_str.strip():
            for p in pr_str.split("|"):
                p = p.strip()
                if p:
                    prereq_set.add(p)

        if md.get("difficulty_level") is not None:
            diffs.append(md["difficulty_level"])
        if md.get("learning_time_days") is not None:
            times.append(md["learning_time_days"])

    diff_summary = None
    if diffs:
        diff_summary = {"avg": round(sum(diffs) / len(diffs), 2), "min": min(diffs), "max": max(diffs)}

    time_summary = None
    if times:
        time_summary = {"avg_days": round(sum(times) / len(times), 2), "min_days": min(times), "max_days": max(times)}

    return {
        "topic": topic,
        "top_k_skills": top_skills[:top_k],
        "prerequisites": sorted(prereq_set),
        "difficulty_summary": diff_summary,
        "learning_time_summary": time_summary,
        "raw_snippets": [n.node.text for n in nodes[:top_k]],
    }
