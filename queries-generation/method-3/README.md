# Call Transcript Analysis & Dataset Generation Pipeline

A comprehensive machine learning pipeline for curating, clustering, and generating analytical datasets from call center transcripts. This project implements advanced sampling techniques and query generation for business intelligence applications.

## 📋 Project Overview

This repository contains three interconnected notebooks that form a complete pipeline for:

1. **Dataset Curation** - Intelligent sampling using FPS (Farthest Point Sampling)
2. **Semantic Clustering** - Intent-based grouping with embedding-based diversity selection
3. **Query Generation** - Automated creation of analytical queries with multi-reasoning types

## 🏗️ Architecture

### Pipeline Components

```
Raw Transcripts
    ↓
[dataset-generation-part-1.ipynb]
    ├─→ FPS Sampling per Domain
    ├─→ K-Means Clustering
    └─→ Visual Analytics
    ↓
Curated Dataset
    ↓
[summary_clusters.ipynb]
    ├─→ Domain/Intent Clustering
    ├─→ Embedding-based Selection
    └─→ Outlier Detection
    ↓
Intent Clusters
    ↓
[dataset-generation-pipeline.ipynb]
    ├─→ Multi-type Query Generation
    ├─→ Follow-up Creation
    └─→ Quality Validation
    ↓
Final Query Dataset
```

## 📚 Notebooks

### 1. dataset-generation-part-1.ipynb

**Purpose**: Curate representative transcripts using cluster-based Farthest Point Sampling (FPS).

**Key Features**:

-   **Automatic K-Tuning**: Finds optimal cluster count per domain using silhouette analysis
-   **FPS Selection**: Selects 4 diverse samples per cluster (centroid, outlier, 2 variants)
-   **Triple Visualization**:
    -   Elbow plot (inertia)
    -   Silhouette score analysis
    -   2D PCA strategy map with convex hulls
-   **Per-Domain Processing**: Handles heterogeneous call center data

**Configuration**:

```python
SAMPLES_PER_CLUSTER = 4    # Centroid + Outlier + 2 Variants
MIN_K = 3                   # Minimum clusters per domain
MAX_K = 10                  # Maximum clusters per domain
```

**Output**:

-   `final_curated_dataset.json` - Selected transcripts (FPS-sampled)
-   `curation_report_metadata.csv` - Selection rationale per transcript

### 2. summary_clusters.ipynb

**Purpose**: Group transcripts by domain and intent, then select diverse representatives using semantic embeddings.

**Key Features**:

-   **Robust Data Loading**: Handles corrupted/truncated JSON with backtracking repair
-   **Local Embeddings**: Uses `sentence-transformers/all-MiniLM-L6-v2` (no API calls)
-   **Fallback Architecture**: Works without LangChain dependencies
-   **Cosine Distance Selection**: Picks 11 transcripts per run (1 anchor + 10 farthest)
-   **15 Runs per Domain**: Ensures comprehensive coverage

**Model Loading**:

```python
# Tries in order:
1. langchain_huggingface.HuggingFaceEmbeddings
2. langchain_community.embeddings.HuggingFaceEmbeddings
3. langchain.embeddings.HuggingFaceEmbeddings
4. NativeHuggingFaceEmbeddings (custom fallback)
```

**Output**:

-   `clusters_final_local.json` - Structured as `{domain: [runs]}` with intent-based selections

### 3. dataset-generation-pipeline.ipynb

**Purpose**: Generate diverse analytical queries with ground-truth answers using LLM-based generation.

**Key Features**:

-   **Multi-Key API Management**: Automatic rotation across 5 Groq API keys with rate limit handling
-   **8 Query Types**:
    1. **Single Transcript** (40 queries) - Single-document analysis
    2. **Cluster Aggregate** (30 queries) - Multi-transcript patterns
    3. **Cross-Cluster** (20 queries) - Compare intents within domain
    4. **Cross-Domain** (15 queries) - Compare across business domains
    5. **Temporal** (10 queries) - Time-based analysis
    6. **Causal Chain** (10 queries) - Multi-hop reasoning
    7. **Counterfactual** (10 queries) - What-if scenarios
    8. **Unanswerable** (20 queries) - Insufficient data cases
-   **Follow-up Generation** (50 queries) - Contextual multi-turn queries
-   **9 Subcategories**: Business Outcomes, Sentiment, Agent Behavior, Process Breakdowns, etc.
-   **Quality Validation**: DeepEval metrics (Faithfulness, Correctness, BERT-Score, ROUGE)

**Reasoning Types**:

-   Causal
-   Temporal
-   Comparative
-   Conditional
-   Behavioral

**Difficulty Levels**:

-   **Easy**: Single observable pattern
-   **Medium**: 2 dimensions + quantitative/temporal elements
-   **Hard**: Multi-hop logic with conditional/causal chains

**Output**:

-   `final_dataset_with_followups.json` - Complete query dataset with answers
-   Quality metrics per query

## 🔧 Installation

### Prerequisites

```bash
# Python 3.8+
pip install numpy pandas matplotlib seaborn
pip install sentence-transformers scikit-learn scipy
pip install tqdm spacy groq
pip install deepeval bert-score rouge-score textstat

# Spacy model
python -m spacy download en_core_web_sm
```

### Optional (for LangChain integration)

```bash
pip install -U langchain-community
pip install langchain-huggingface
```

## 🚀 Usage

### Step 1: Curate Dataset

```python
# In dataset-generation-part-1.ipynb
DATA_PATH = './final-transcripts-domain-corrected.json'
run_pipeline()
```

**Outputs**:

-   `final_curated_dataset.json` (FPS-sampled transcripts for Stage 1)
-   `curation_report_metadata.csv`
-   Visualization graphs per domain

### Step 2: Create Intent Clusters

```python
# In summary_clusters.ipynb
INPUT_FILE = "../../transcript-rag/const/summaries-20k.json"
OUTPUT_FILE = "clusters_final_local.json"
run_pipeline()
```

**Outputs**:

-   `clusters_final_local.json` (Intent-based clusters for Stage 2)

### Step 3: Generate Queries

```python
# In dataset-generation-pipeline.ipynb
# Upload both files when prompted:
# 1. final_curated_dataset.json (FPS-sampled transcripts from Step 1)
# 2. clusters_final_local.json (Intent clusters from Step 2)

# Enter 1-5 Groq API keys when prompted
# Pipeline runs automatically
```

**Outputs**:

-   `final_dataset_with_followups.json`

## 📊 Key Algorithms

### Farthest Point Sampling (FPS)

Ensures maximum diversity within each cluster:

```python
def farthest_point_sampling(embeddings, n_samples, centroid_idx):
    selected = [centroid_idx]  # Start with centroid

    for i in range(n_samples - 1):
        dists = pairwise_distances(embeddings, embeddings[selected])
        min_dists = np.min(dists, axis=1)
        next_idx = np.argmax(min_dists)  # Farthest from all selected
        selected.append(next_idx)

    return selected
```

### Automatic K-Tuning

Uses silhouette score to find optimal cluster count:

```python
def find_optimal_k_with_inertia(embeddings, domain_name):
    sil_scores = []
    k_range = range(MIN_K, MAX_K + 1)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE)
        labels = kmeans.fit_predict(embeddings)
        score = silhouette_score(embeddings, labels)
        sil_scores.append(score)

    best_k = k_range[np.argmax(sil_scores)]
    return best_k
```

### Multi-Key API Management

Handles rate limits across multiple API keys:

```python
class MultiKeyGroqModel:
    def _switch_api_key(self):
        # Rotate to next key on rate limit
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        self.client = Groq(api_key=self.api_keys[self.current_key_index])

    def generate(self, prompt, json_mode=False):
        # Automatic retry with exponential backoff
        # Switches keys after 3 failures
        # Returns JSON or text response
```

## 📈 Visualization Outputs

### dataset-generation-part-1.ipynb

For each domain, generates 3 graphs:

1. **Elbow Plot**: Shows inertia (within-cluster variance) vs k
2. **Silhouette Analysis**: Shows clustering quality score vs k
3. **Strategy Map**: 2D PCA with:
    - Convex hulls per cluster (colored)
    - ⭐ Centroids (large stars)
    - ❌ Outliers (X markers)
    - 🔺 Variants (triangles)

### Example Cluster Visualizations by Domain

The pipeline automatically generates clustering visualizations for each domain based on `domain + intent + reason_for_call` features. Below are representative examples across all 6 domains:

#### Hotel Domain (k=6 clusters)

![Hotel Clustering](images/hotel_clusters.png)

-   **Optimal k=6** selected via silhouette analysis (score: 0.19)
-   Clusters represent: reservations, cancellations, room service, amenities, billing, loyalty programs
-   6 distinct guest service patterns with moderate overlap between booking-related intents

#### Banking Domain (k=6 clusters)

![Banking Clustering](images/banking_clusters.png)

-   **Optimal k=6** selected via silhouette analysis (score: 0.23)
-   Highest silhouette score indicates well-separated clusters
-   6 distinct customer intent patterns identified
-   Clear separation between account inquiries, fraud reports, loan requests, and payment disputes

#### Flight Domain (k=4 clusters)

![Flight Clustering](images/flight_clusters.png)

-   **Optimal k=4** selected (silhouette score: 0.196)
-   Clusters represent: booking changes, cancellations, baggage issues, loyalty programs
-   Tighter cluster boundaries indicate more homogeneous intents
-   Large beige cluster shows high volume of basic booking modifications

#### Insurance Domain (k=7 clusters)

![Insurance Clustering](images/insurance_clusters.png)

-   **Optimal k=7** selected (silhouette score: 0.18)
-   Most diverse domain with 7 distinct patterns
-   Includes claims processing, policy inquiries, premium questions, coverage disputes
-   Red cluster (top) shows isolated high-complexity claims

#### Retail Domain (k=4 clusters)

![Retail Clustering](images/retail_clusters.png)

-   **Optimal k=4** selected (silhouette score: 0.205)
-   Clusters: order tracking, returns/refunds, product inquiries, delivery issues
-   Balanced cluster sizes with clear convex hull boundaries
-   Red cluster (top) represents high-value customer escalations

#### Telecom Domain (k=4 clusters)

![Telecom Clustering](images/telecom_clusters.png)

-   **Optimal k=4** selected (silhouette score: 0.194)
-   Patterns: billing disputes, service outages, plan changes, technical support
-   Large lower cluster indicates high volume of technical support calls
-   Cyan cluster (top left) shows proactive service upgrade inquiries

### Interpretation Guide

**Elbow Plot (Left)**:

-   Steep drops indicate meaningful cluster additions
-   Flattening curve suggests optimal k reached
-   Used as secondary validation metric

**Silhouette Analysis (Middle)**:

-   Higher scores = better-defined clusters (0.2+ is good for real-world text data)
-   Red dashed line marks the selected optimal k
-   Peak indicates best separation between clusters

**FPS Strategy Map (Right)**:

-   Each color represents one cluster
-   Convex hulls show cluster boundaries in 2D PCA space
-   Selected samples (stars, X's, triangles) ensure diversity coverage
-   Overlapping regions indicate semantic similarity between intents

## 🎯 Query Generation Strategy

### Task 1: Primary Queries (155 total)

| Type              | Count | Scope                    |
| ----------------- | ----- | ------------------------ |
| Single Transcript | 40    | Individual call analysis |
| Cluster Aggregate | 30    | Multi-call patterns      |
| Cross-Cluster     | 20    | Intent comparison        |
| Cross-Domain      | 15    | Domain comparison        |
| Temporal          | 10    | Time-based patterns      |
| Causal Chain      | 10    | Multi-hop reasoning      |
| Counterfactual    | 10    | What-if scenarios        |
| Unanswerable      | 20    | Insufficient data        |

### Task 2: Follow-up Queries (50 total)

-   60% probability of follow-up generation
-   1-5 follow-up levels per primary query
-   Context-aware refinement/clarification

## 🔍 Quality Metrics

Evaluated using DeepEval:

-   **Faithfulness**: Answer grounded in transcript
-   **Answer Correctness**: Semantic accuracy
-   **BERT-Score**: Token-level similarity
-   **ROUGE-L**: N-gram overlap

## 📂 Data Structure

### Transcript Object

```json
{
    "transcript_id": "TRANS_12345",
    "domain": "Banking",
    "intent": "Account_Inquiry",
    "reason_for_call": "Check balance and recent transactions",
    "summary": "Customer called to verify account balance...",
    "full_transcript": "Agent: Hello, how can I help you today?..."
}
```

### Query Object

```json
{
  "query_id": "Q_001",
  "query": "What caused the customer to escalate their request?",
  "answer": "The customer escalated because...",
  "reasoning_type": "causal",
  "subcategory": "Customer Frustration & Sentiment Drivers",
  "difficulty": "medium",
  "transcript_ids": ["TRANS_12345"],
  "domain": "Banking",
  "task_type": "single_transcript",
  "followups": [...]
}
```

## ⚙️ Configuration

### Global Settings

```python
# dataset-generation-part-1.ipynb
RANDOM_STATE = 42
SAMPLES_PER_CLUSTER = 4
MIN_K = 3
MAX_K = 10

# summary_clusters.ipynb
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RUNS_PER_DOMAIN = 15
SAMPLES_PER_RUN = 11

# dataset-generation-pipeline.ipynb
FOLLOWUP_PROBABILITY = 0.6
MAX_FOLLOWUP_DEPTH = 5
```

## 🛠️ Troubleshooting

### Issue: JSON Corrupted Error

**Solution**: `summary_clusters.ipynb` includes automatic repair:

```python
def load_data_robust(filepath):
    # Attempts backtracking repair on corrupted JSON
    # Recovers up to last valid object
```

### Issue: API Rate Limits

**Solution**: Use multiple API keys (up to 5 supported):

```python
# Automatic key rotation
# 60s backoff on single key
# Switches after 3 consecutive failures
```

### Issue: LangChain Import Errors

**Solution**: Native fallback automatically activates:

```python
class NativeHuggingFaceEmbeddings:
    # Direct sentence-transformers usage
    # No LangChain dependency
```

## 📊 Expected Performance

-   **Dataset Curation**: ~100-200 transcripts from 1000+ inputs
-   **Cluster Generation**: 15 runs × 11 samples per domain
-   **Query Generation**: ~205 total queries (155 primary + 50 follow-ups)
-   **Processing Time**:
    -   Step 1: 5-15 min (depends on data size)
    -   Step 2: 10-20 min (local embeddings)
    -   Step 3: 30-60 min (LLM API calls)

## 🎓 Use Cases

1. **Call Center Analytics**: Identify service patterns and improvement areas
2. **Training Data Generation**: Create QA pairs for RAG systems
3. **Business Intelligence**: Extract insights from unstructured conversations
4. **Model Fine-tuning**: Generate diverse reasoning tasks
5. **Quality Assurance**: Benchmark agent performance patterns

## 📄 License

This project is provided as-is for research and commercial use.

## 🤝 Contributing

For improvements or bug reports, please submit issues with:

-   Notebook name
-   Error message/behavior
-   Data sample (if applicable)

## 📧 Contact

For questions about implementation or customization, refer to the detailed code comments in each notebook.

---

**Built with**: Python, scikit-learn, sentence-transformers, Groq API, DeepEval

**Last Updated**: December 2025
