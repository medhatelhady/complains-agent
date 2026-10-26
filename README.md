# Customer Complaints: Topic Discovery and Operations Agent

This project takes a bank's consumer complaints (a CFPB-style export) and turns them into a tool the operations team can use. The steps are:

1. Clean and explore the complaint narratives.
2. Find recurring topics with clustering, then merge the clusters into a 12-topic taxonomy.
3. Train a classifier so that new complaints can be assigned a topic without re-running the clustering.
4. Run an LLM-powered operations agent on each new complaint. It classifies the complaint, finds similar past cases, checks the risk, triages it and drafts a case note and a customer reply. A person stays in control throughout.

## Project structure

```
.
├── notebooks/
│   ├── 01_data_analysis.ipynb      # gather, assess, clean, EDA
│   ├── 02_clustering.ipynb         # TF-IDF + embeddings, clustering, taxonomy
│   ├── 03_topic_classifier.ipynb   # supervised model on the K-Means clusters
│   └── 04_operation_agent.ipynb    # LangGraph agent + weekly ops briefing
├── data/
│   ├── raw/complaints_flat.csv     # input data
│   ├── clean/                      # output of 01
│   ├── interim/                    # outputs of 02
│   └── output/                     # outputs of 04
├── models/                         # trained classifier and Chroma vector store
├── docs/                           # images used in this README
├── requirements.txt
└── .env                            # API credentials (not committed)
```

## Setup

You need **Python 3.13**.

```bash
python -m pip install -r requirements.txt
```

Create a `.env` file in the project root. Notebooks 02 and 04 call an OpenAI-compatible LLM (`gpt-4o-mini`):

```
OPENAI_API_KEY=your-key
OPENAI_BASE_URL=https://api.openai.com/v1
```

Put the raw data at `data/raw/complaints_flat.csv`.

## How to run

Run the notebooks in order. Each one reads what the previous one saved.

| # | Notebook | Reads | Writes |
|---|----------|-------|--------|
| 1 | `01_data_analysis` | `data/raw/complaints_flat.csv` | `data/clean/complaints_clean.csv` |
| 2 | `02_clustering` | `complaints_clean.csv` | `data/interim/`: `complaints_clusters.csv`, `embeddings.npy`, `taxonomy.json`, `cluster_to_topic.json`, `cluster_names.json`, `cluster_summary.csv`, `method_comparison.csv` |
| 3 | `03_topic_classifier` | `complaints_clean.csv`, `complaints_clusters.csv` | `models/kmeans_cluster_classifier.joblib` |
| 4 | `04_operation_agent` | all of the above | `models/chroma_db/`, `data/output/agent_results.csv`, `data/output/ops_briefing.md` |

## What each notebook does

### 01 · Data analysis
- Assesses data quality. Many columns are mostly empty, only about 27% of complaints have a narrative, product names are duplicated and some complaints appear more than once.
- Cleans the data. It drops uninformative columns, merges old and new product names, keeps only complaints that have text, removes texts under 15 words and duplicates, and adds a `clean_text` column for modelling.
- EDA: after cleaning there are **20,551 complaints** from March 2015 to March 2021. Credit/prepaid cards, checking/savings and mortgages make up about 80% of them. The issue each customer picks from the drop-down is vague, which is why the topics are taken from the text instead.

### 02 · Clustering and taxonomy
- Represents each complaint with **TF-IDF** (words) and **sentence embeddings** (`all-MiniLM-L6-v2`, meaning), combines the two and reduces them with **UMAP**.
- Compares K-Means, Agglomerative (Ward), HDBSCAN and an NMF baseline using silhouette, agreement between methods (ARI) and seed stability.
- An LLM drafts a name for each K-Means cluster. The clusters are then merged by hand into **12 operational topics**, each with a one-line definition (`taxonomy.json`). The cluster-to-topic mapping is saved in `cluster_to_topic.json`.

### 03 · Topic classifier
- Inputs are `clean_text`, `tags`, `submitted_via` and `n_words`. Product and issue are deliberately left out.
- One scikit-learn pipeline holds the features (TF-IDF, one-hot, scaled length) and the model. It compares logistic regression, linear SVM and Complement Naive Bayes with 5-fold CV, then tunes the best of them with a grid search. The 20% test set is used only once, at the end.
- The whole pipeline is saved as a single joblib file, which takes raw columns and returns the cluster.
- Note: the classifier is scored on how well it reproduces K-Means. There are no human labels to measure accuracy against.

### 04 · Operations agent
The agent is a **LangGraph** workflow. The three tools run in parallel, and triage waits for all of them:

<p align="center">
  <img src="docs/agent_graph.png" alt="Agent graph: classify, similar_cases and check_risk run in parallel, then triage, draft, save" width="360">
</p>

1. **Classify.** It predicts the cluster, maps it to one of the 12 topics and gives a confidence score. The review threshold is calibrated on the held-out test set, so the team can trade coverage against accuracy.
2. **Similar cases (RAG).** It retrieves the closest past complaints, and how they ended, from a **Chroma** vector store through LangChain.
3. **Risk check (LLM).** The LLM gives a structured yes/no answer with a reason. It is told about vulnerable-customer tags, and a failed call counts as "yes".
4. **Triage.** Fixed rules choose the owning team, the target days and the priority, and decide whether a person must review the case.
5. **Draft and check.** The LLM writes a structured case note and customer reply. Plain code then checks the draft. A failed check flags the case for a person instead of blocking it.
6. **Save.** Everything goes to `agent_results.csv`.

The notebook also produces a **weekly operations briefing** (`ops_briefing.md`). It shows the volume per topic over the last 4 weeks against the 12 weeks before, the topics that are rising, the high-risk complaints to look at first, and the cases sent to people.

**People stay in control at three points:**
- **Uncertain topic:** below the confidence threshold the graph pauses (`interrupt`) until a reviewer picks the topic.
- **High risk:** the case becomes High priority and goes to a person, along with the reason.
- **Drafts only:** nothing is sent to a customer automatically.

The demo runs four cases: a clear case, a vague case, a high-risk case and a complaint that tries to steer the agent (prompt injection).

## Limitations
- Every label comes from AI (clustering plus an LLM-assisted taxonomy), so the reported accuracy is agreement with that labelling, not ground truth.
- The clusters are mixed, so the cluster-to-topic mapping is approximate. Auto loan & finance is the weakest topic.
- The routing table (topic → team and target days) is a placeholder, to be agreed with the operations lead.
