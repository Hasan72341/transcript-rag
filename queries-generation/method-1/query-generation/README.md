<h1 align="center">QUERY GENERATION</h1>

### Requirements
- update the `GROQ_API_KEY` in the file.

### Pipeline Components
- To generate the queries, run `query-generator-task-1-2.ipynb` file.

```
1. Domain
2. No of queries
3. Difficulty
4. mece category (optional)
5. sub-category (optional)
    ↓
[stage-1, query-generator-task-1-2.ipynb]
    └─→ Intents, and Bussiness Events
    ↓
[stage-2, query-generator-task-1-2.ipynb]
    └─→ Task 1 - Queries
    ↓
[stage-3, query-generator-task-1-2.ipynb]
    └─→ Task 2 - Queries
```
