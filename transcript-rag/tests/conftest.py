import os

# Avoid nested OpenMP pools when FAISS runs inside LangGraph's worker threads.
os.environ.setdefault("OMP_NUM_THREADS", "1")
