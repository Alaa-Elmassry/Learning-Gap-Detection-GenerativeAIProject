# Learning Gap Detection

> Gradio application that generates MCQ questions for a user-entered topic, evaluates the answers, detects missing prerequisites, then produces a personalized learning roadmap.

---

## Demo

- **Demo video:** [My Demo video link here](https://drive.google.com/file/d/1RRZx3zfNAR95O9RbugE8QveNj1vMG-hk/view?usp=sharing)


---

## Project Overview

This project is a **Learning Gap Detection** system with a Gradio interface.

The user enters a topic such as `SQL`, `AI`, or `Docker`, then the application:

1. retrieves related context using **RAG**,
2. generates **5 to 7 MCQ questions**,
3. accepts the user's answers,
4. analyzes performance,
5. identifies weak concepts and missing prerequisites,
6. generates a personalized roadmap to close the learning gaps,
7. saves the full session as a JSON file inside `outputs/`.

The UI title inside the app is:

**Learning Gap Detection (MCQ + Roadmap)**

---

## What the Project Uses

### 1) ChromaDB
**Role in this project:** persistent vector storage.

The project stores embeddings in a persistent Chroma database folder called:

- `chroma_db`

Behavior from the code:
- If `chroma_db` already contains vectors, the application loads the existing index.
- If it does not contain vectors yet, the application reads the dataset and builds the vector index automatically.

So in this project, **ChromaDB is not manually required every time**. It is created automatically on first build if it does not already exist.

### 2) LlamaIndex
**Role in this project:** orchestration layer for the retrieval pipeline.

It is used to:
- convert dataset rows into documents,
- connect to the Chroma vector store,
- build the `VectorStoreIndex`,
- retrieve the most relevant nodes for a topic.

### 3) RAG (Retrieval Augmented Generation)
**Role in this project:** supplying the model with retrieved context before generation.

The retrieved context includes fields such as:
- topic
- top skills
- prerequisites
- difficulty summary
- learning time summary
- raw text snippets

This context is then inserted into prompts for:
- question generation,
- answer analysis,
- roadmap generation.

### 4) Hugging Face Inference API
**Role in this project:** model inference for structured JSON generation.

The project uses a custom `HFClient` wrapper around `InferenceClient`.

**Main generation model used in the code:**
- `mistralai/Mistral-7B-Instruct-v0.2`

**Embedding model used in the code:**
- `sentence-transformers/all-MiniLM-L6-v2`

---

## How the Project Works

## Flow of the Project

### Step 1 - Application startup
When `app.py` runs, it does the following first:

- loads environment variables using `python-dotenv`,
- creates the `outputs/` folder if it does not exist,
- creates or loads the RAG index using `build_or_load_index(...)`,
- initializes the Hugging Face client using `HFClient(...)`.

### Step 2 - User enters a topic
The user types a topic in the textbox, for example:
- SQL
- AI
- Docker

Then the user clicks:
- **Generate MCQ**

### Step 3 - Retrieve context for the topic
The project calls the retriever and builds a context object for the topic.

That context contains:
- top related skills,
- merged prerequisites,
- difficulty summary,
- learning time summary,
- retrieved snippets.

### Step 4 - Generate MCQ questions
The project sends the retrieved context to the Hugging Face model through the `questions_prompt(...)` prompt.

Question generation rules enforced by the code and prompt:
- questions are **MCQ only**,
- total question count must be **5 to 7**,
- each question must have **exactly 4 options**,
- `answer_key` must be one of `A`, `B`, `C`, `D`.

The project also normalizes the result locally to force the MCQ schema.

### Step 5 - Show questions in the UI
The generated questions are shown in the Gradio interface.

For each question:
- the question text is displayed,
- the 4 options are displayed,
- the user selects one answer from a dropdown using `A / B / C / D`.

### Step 6 - Submit answers
After answering, the user clicks:
- **Submit**

The application then:
- collects the selected answers,
- sends the full context + questions + user answers to the analysis prompt,
- asks the model to return structured JSON analysis.

### Step 7 - Grade and analyze
The system performs two things:

#### A. Local grading
It compares the user's answer with `answer_key` for each question and computes:
- correct count,
- total count,
- percentage score,
- per-question feedback.

#### B. Model-based learning-gap analysis
The analysis prompt asks the model to return:
- overall score

### Step 8 - Generate roadmap
The project automatically collects roadmap topics from:
- `missing_prerequisites`,
- `weak_concepts_overall`,
- detected question concepts.

Then for each selected roadmap topic, it:
- retrieves fresh context from the index,
- sends the context and analysis to the roadmap prompt,
- generates a roadmap.

The roadmap logic also tries to guarantee at least **3 course**.
If the model returns too few steps, the code tries to:
1. extend the roadmap,
2. regenerate it,
3. fill missing steps,
4. fill missing resource buckets if needed.

### Step 9 - Save session output
After analysis and roadmap generation, the project saves a JSON session file in:

- `outputs/session_<timestamp>.json`

This file includes:
- timestamp,
- context,
- generated questions,
- user answers,
- analysis,
- roadmap.

### Step 10 - Show final result
The UI displays:
- results and score,
- roadmap.

The roadmap renderer also builds a Mermaid flow/graph view when rendering the roadmap output.

---

## Is the Project Dynamic or Static?

This project is **mostly dynamic**.

### Dynamic parts
These parts are generated dynamically at runtime:
- retrieved RAG context based on the entered topic,
- generated questions,
- answer analysis,
- weak concepts and missing prerequisites,
- roadmap topics,
- roadmap steps,
- curated resources inside roadmap steps.

### Fixed / constrained parts
These parts are fixed by the code:
- the UI structure,
- the dataset path: `data/skills_dataset.xlsx`,
- the Hugging Face generation model name,
- MCQ format rules,
- answer letters limited to `A / B / C / D`,
- question count constrained to `5-7`.

So the project is **not hardcoded in output content**, but it does have **fixed rules, fixed model names, and fixed file paths**.

---

## Dataset and Index Behavior

### Dataset path used in the code
- `data/skills_dataset.xlsx`

### Chroma directory used in the code
- `chroma_db`

### Important behavior
This is the exact practical behavior of the project:

- If `chroma_db` already has stored vectors, the project loads them.
- If `chroma_db` is missing or empty, the project reads `data/skills_dataset.xlsx` and builds the index automatically.

That means the README should treat the system like this:

- **dataset is required for first-time index creation**,
- **prebuilt ChromaDB is enough if it already exists and contains vectors**.

---

## Expected Dataset Fields

From the retrieval pipeline, the dataset rows are transformed using these fields:

- `skill_name`
- `category`
- `skill_type`
- `difficulty_level`
- `learning_time_days`
- `job_demand_score`
- `salary_impact_percent`
- `market_trend`
- `future_relevance_score`
- `prerequisites/0`
- `prerequisites/1`
- `prerequisites/2`
- `prerequisites/3`
- `prerequisites/4`

The prerequisites are merged into one prerequisites field inside the retrieved context.

---

## Project Files

Core files used by the app:

- `app.py` - main Gradio application and full workflow
- `rag_pipeline.py` - dataset-to-index pipeline and retrieval logic
- `hf_client.py` - Hugging Face JSON generation wrapper
- `prompts.py` - prompts for questions, analysis, and roadmap
- `build_index.py` - optional helper to build the index directly

Runtime folders:

- `data/` - contains the Excel dataset
- `chroma_db/` - persistent vector database
- `outputs/` - saved session results

---

## Requirements to Run the Project

The project needs these things to run correctly:

1. Python environment with the required packages installed.
2. A Hugging Face token in environment variables.
3. One of the following:
   - a ready `chroma_db/` folder with stored vectors,
   - or `data/skills_dataset.xlsx` so the index can be created automatically.

### Required environment variable
- `HF_TOKEN`

### Optional environment variable
- `HF_PROVIDER`

The code defaults to:
- `featherless-ai`

If `HF_PROVIDER=auto`, the code still converts it to:
- `featherless-ai`

---

## How to Run the Project Locally

### 1) Activate your virtual environment
Example on Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 2) Install the dependencies
```bash
pip install -r requirements.txt
```

### 3) Make sure your files are available
You need either:

- `chroma_db/`

or:

- `data/skills_dataset.xlsx`

If `chroma_db/` does not exist yet, the project will create it automatically from the dataset during startup.

### 4) Set environment variables
Example:

```powershell
$env:HF_TOKEN="your_token_here"
$env:HF_PROVIDER="featherless-ai"
```

### 5) Run the app
```bash
python app.py
```

### 6) Open the local Gradio link
By default, the app runs on a local Gradio URL such as:

```text
http://127.0.0.1:7860
```

---

## Optional: Build the Index Separately
If you want to build the vector index before starting the app, run:

```bash
python build_index.py
```

This calls the same `build_or_load_index(...)` pipeline used by the main app.

---

## UI Summary

The interface contains:
- Topic textbox
- **Generate MCQ** button
- Questions accordion
- **Submit** button
- Analysis accordion
- Roadmap accordion
- **New Topic / Reset** button

---

## Prompt Design Used in the Project

The project defines three main prompts:

### 1) `questions_prompt(...)`
Used to generate the MCQ questions.

### 2) `analysis_prompt(...)`
Used to analyze user answers and identify learning gaps.

### 3) `roadmap_prompt(...)`
Used to generate a personalized roadmap with resources.

All prompts require **strict JSON output only**.

---

## Output Files

Each completed run saves a file inside:

- `outputs/`

File format:

```text
outputs/session_<timestamp>.json
```

Saved data includes:
- context,
- questions,
- user answers,
- analysis,
- roadmap.

---

## Notes

- The project creates `outputs/` automatically.
- The project creates or loads `chroma_db/` automatically.
- If the model returns fewer than 5 questions, the app asks the user to generate again.
- The roadmap code has extra handling to avoid returning too few steps.
- The output content is dynamic, but the workflow and constraints are fixed by code.



