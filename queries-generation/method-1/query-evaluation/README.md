<h1 align="center">QUERY EVALUATION</h1>

### Requirements
- update the `GROQ_API_KEY` or `API_KEYS` in the respective files.

### Pipeline Components

```
Intents, and Bussiness Events (from query-generation)
    ↓
[stage-1]
    └─→ [llm-eval-stage-1.ipynb]
```

```
Task 1 - Queries (task1.json)
    ↓
[stage-2]
    └─→ [llm-as-a-judge-task-1.ipynb, non-llm-eval-task-1.ipynb]
```

```
Task 2 - Queries (task2.json)
    ↓
[stage-3]
    └─→ [llm-as-a-judge-task-2.ipynb, non-llm-eval-task-2.ipynb]
```


