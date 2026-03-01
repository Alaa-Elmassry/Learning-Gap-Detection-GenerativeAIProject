from rag_pipeline import RAGConfig, build_or_load_index

if __name__ == "__main__":
    cfg = RAGConfig()
    _ = build_or_load_index(cfg)
    print("Index ready.")
