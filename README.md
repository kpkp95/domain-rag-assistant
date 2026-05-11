# Domain RAG Assistant

A source-grounded Retrieval-Augmented Generation app that answers questions from domain-specific documents using LangChain, Gemini, Pinecone, Flask, Docker, and AWS.

The first working version is built as a medical document assistant. The project is designed to later support multiple knowledge domains such as medical documents, machine learning books, LLM books, and course notes.

---

## Project Overview

Domain RAG Assistant is an end-to-end document-based chatbot. It allows users to ask questions and receive answers grounded in uploaded PDF documents.

Instead of relying only on the general knowledge of an LLM, the app retrieves relevant document chunks from a vector database and gives the LLM only that retrieved context. This makes the answers more specific, source-based, and easier to verify.

The current version uses medical reference documents as the first domain.

---

## Problem Statement

Large language models can generate helpful answers, but they can also hallucinate or answer from outdated/general knowledge.

This project solves that problem by building a RAG pipeline where:

1. Domain documents are loaded from PDF files.
2. The documents are split into smaller chunks.
3. Each chunk is converted into embeddings.
4. Embeddings are stored in Pinecone.
5. User questions are matched with the most relevant chunks.
6. Gemini generates an answer using only the retrieved context.
7. The app returns the answer with source citations.

---

## Key Features

- PDF document ingestion
- Text chunking with LangChain
- HuggingFace sentence-transformer embeddings
- Pinecone vector database
- Similarity-based document retrieval
- Gemini-powered answer generation
- Flask web application
- Source citations with file, page, domain, and chunk ID
- Medical-domain RAG assistant
- Domain-ready metadata design
- Improved chat UI
- Loading indicator
- Clear chat button
- Dockerized application
- AWS ECR deployment
- AWS EC2 deployment
- GitHub Actions CI/CD
- Self-hosted GitHub Actions runner on EC2

---

## Current Version

The current version supports the medical domain.

```txt
Domain = medical
Data folder = data/medical
Pinecone index = domain-rag-assistant
```

Each stored chunk includes metadata:

```json
{
  "source": "medical_book.pdf",
  "page": 45,
  "domain": "medical",
  "chunk_id": 102
}
```

This metadata is used for source citations and will also support future domain switching.

---

## Tech Stack

| Layer                | Technology                             |
| -------------------- | -------------------------------------- |
| Programming Language | Python                                 |
| Backend              | Flask                                  |
| LLM                  | Gemini                                 |
| RAG Framework        | LangChain                              |
| Embeddings           | sentence-transformers/all-MiniLM-L6-v2 |
| Vector Database      | Pinecone                               |
| Frontend             | HTML, CSS, JavaScript, Bootstrap       |
| Containerization     | Docker                                 |
| CI/CD                | GitHub Actions                         |
| Cloud Registry       | AWS ECR                                |
| Deployment           | AWS EC2                                |
| Runner               | GitHub self-hosted runner              |

---

## Architecture

```txt
PDF Documents
     |
     v
LangChain PDF Loader
     |
     v
Text Cleaning + Metadata Preservation
     |
     v
Recursive Text Splitting
     |
     v
HuggingFace Embeddings
     |
     v
Pinecone Vector Database
     |
     v
Retriever
     |
     v
Retrieved Context + User Question
     |
     v
Gemini LLM
     |
     v
Answer + Source Citations
     |
     v
Flask Web UI
```

---

## RAG Workflow

### 1. Load Documents

PDF files are loaded from the selected domain folder.

Example:

```txt
data/medical/
```

### 2. Preserve Metadata

For each page, the app keeps useful metadata:

```txt
source
page
domain
```

### 3. Split Text into Chunks

Large documents are split into smaller chunks using LangChain’s `RecursiveCharacterTextSplitter`.

Current chunk settings:

```python
chunk_size = 500
chunk_overlap = 20
```

### 4. Add Chunk IDs

Each chunk gets a unique `chunk_id`.

This helps with citation tracking.

### 5. Create Embeddings

The app uses:

```txt
sentence-transformers/all-MiniLM-L6-v2
```

This model creates vectors with dimension:

```txt
384
```

So the Pinecone index is created with:

```txt
dimension = 384
metric = cosine
```

### 6. Store in Pinecone

Document chunks and metadata are uploaded to Pinecone.

### 7. Retrieve Relevant Chunks

When the user asks a question, Pinecone retrieves the most relevant chunks.

Current setting:

```python
k = 5
```

### 8. Generate Answer with Gemini

Gemini receives:

```txt
User question
+
Retrieved document context
```

The prompt instructs the model to answer only from the provided context.

### 9. Return Sources

The app returns both:

```txt
answer
sources
```

Example source:

```txt
medical_book.pdf — page 45 — chunk 102
```

---

## Medical Disclaimer

This chatbot is for educational and document-based information only.

It does not provide medical diagnosis, treatment decisions, or emergency advice. Users should consult a qualified healthcare professional for serious or urgent medical concerns.

---

## Project Structure

```txt
domain-rag-assistant/
│
├── src/
│   ├── __init__.py
│   ├── helper.py
│   └── prompt.py
│
├── research/
│   └── trials.ipynb
│
├── data/
│   ├── medical/
│   ├── machine_learning/
│   └── llm/
│
├── static/
│   ├── style.css
│   └── script.js
│
├── templates/
│   └── chat.html
│
├── evaluation/
│   ├── questions.json
│   ├── evaluate.py
│   └── results.csv
│
├── .github/
│   └── workflows/
│       └── main.yml
│
├── app.py
├── store_index.py
├── setup.py
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .gitignore
├── .env.example
└── README.md
```

---

## Important Files

### `src/helper.py`

Contains reusable helper functions:

- Load PDF files
- Filter metadata
- Split documents into chunks
- Add chunk IDs
- Load embeddings
- Format source citations

### `src/prompt.py`

Contains the system prompt used by the RAG chain.

### `store_index.py`

Loads PDFs, creates chunks, generates embeddings, and uploads them to Pinecone.

Run this only when adding or updating documents.

### `app.py`

Main Flask application.

It connects to Pinecone, creates the retriever, calls Gemini, and returns answers with sources.

### `templates/chat.html`

Frontend chat interface.

### `static/style.css`

Custom UI styling.

### `.github/workflows/main.yml`

GitHub Actions workflow for Docker build, ECR push, and EC2 deployment.

---

## Environment Variables

Create a `.env` file locally:

```env
PINECONE_API_KEY=your_pinecone_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
FLASK_SECRET_KEY=your_flask_secret_key_here
```

Do not commit `.env` to GitHub.

A safe example file is included:

```txt
.env.example
```

---

## Gemini / Vertex setup

This project uses Google's Vertex AI (Gemini). The `GOOGLE_API_KEY` environment variable is used by the Vertex client setup in the app. You can provide credentials in one of two ways:

- Set `GOOGLE_API_KEY` to the path of a service-account JSON key file:

```env
GOOGLE_API_KEY=/path/to/service-account.json
```

- Or configure Application Default Credentials (ADC) with `gcloud auth application-default login` on your machine or by attaching a service account to the compute environment. The service account should have appropriate roles (for example: `Vertex AI User` and any storage roles you need).

Keep credentials private and never commit service-account JSON files to the repository.

---

## Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/domain-rag-assistant.git
cd domain-rag-assistant
```

### 2. Create Conda environment

```bash
conda create -n mediChatbot python=3.10 -y
conda activate mediChatbot
```

### 3. Install requirements

```bash
pip install -r requirements.txt
```

### 4. Add environment variables

Create `.env`:

```env
PINECONE_API_KEY=your_pinecone_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
FLASK_SECRET_KEY=your_flask_secret_key_here
```

### 5. Add PDF files

Place your medical PDF files inside:

```txt
data/medical/
```

To mount local data (useful during development):

```bash
docker run --env-file .env -v $(pwd)/data:/app/data -p 8080:8080 domain-rag-assistant:latest
```

On Windows PowerShell replace `$(pwd)` with `${PWD}`.

---

## CI / Deploy (GitHub Actions → ECR / EC2)

The repository includes a GitHub Actions workflow that builds the Docker image, pushes it to ECR, and performs deployment to EC2. See the workflow for details:

[.github/workflows/cicd.yml](.github/workflows/cicd.yml)

The workflow expects GitHub Secrets to be configured for `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`, and `ECR_REPO`. During deployment the runner or infrastructure should have access to the necessary AWS permissions.

---

## Run tests

Run the included test file with `pytest`:

```bash
pip install -r requirements.txt
pytest -q test_setup.py
# or run the full test suite
pytest
```

If you use a separate test environment, activate it before running tests.

---

## Security & Secrets

- Never commit `.env` or any secret files to version control. This repo includes `.env.example` to show required variables.
- For deployments, store secrets in a secure service such as **AWS Secrets Manager**, **AWS Systems Manager Parameter Store**, or GitHub Secrets for Actions.
- When deploying to EC2, consider using an IAM role attached to the instance instead of embedding long-lived credentials.

Note: you mentioned you have not committed anything — good practice. Keep secrets out of git history and use a secrets manager for production deployments.

````

PDF files are not included in this repository.

### 6. Upload documents to Pinecone

```bash
python store_index.py
````

### 7. Run the Flask app

```bash
python app.py
```

Open:

```txt
http://localhost:8080
```

---

## AWS Deployment

The app is deployed using:

```txt
GitHub Actions
AWS ECR
AWS EC2
Docker
Self-hosted GitHub runner
```

### Deployment Flow

```txt
Push to GitHub main branch
        |
        v
GitHub Actions starts
        |
        v
Build Docker image
        |
        v
Push image to AWS ECR
        |
        v
EC2 self-hosted runner pulls latest image
        |
        v
Old container is stopped and removed
        |
        v
New container starts on port 8080
```

### Required GitHub Secrets

Add these secrets in GitHub:

```txt
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
AWS_REGION
ECR_REPO
PINECONE_API_KEY
GOOGLE_API_KEY
FLASK_SECRET_KEY
```

Example values:

```txt
AWS_REGION=us-east-1
ECR_REPO=domain-rag-assistant
```

### EC2 Setup

The EC2 instance needs:

```txt
Ubuntu
Docker installed
GitHub self-hosted runner configured
Security group allowing port 8080
```

Inbound rule:

```txt
Custom TCP | 8080 | 0.0.0.0/0
```

For this project, a very small EC2 instance may fail because HuggingFace embeddings require memory.

A previous low-memory instance exited with:

```txt
Exited (137)
```

The working deployment used an EC2 instance with 8 GB RAM.

---

## GitHub Actions Workflow

The workflow:

1. Checks out the code
2. Configures AWS credentials
3. Logs in to ECR
4. Builds Docker image
5. Pushes image to ECR
6. Runs deployment on EC2 self-hosted runner
7. Stops old container
8. Pulls latest image
9. Runs new container

---

## Current Deployment Status

The application has been successfully deployed on AWS EC2 using Docker and GitHub Actions CI/CD.

Current deployment:

```txt
AWS EC2 + Docker + AWS ECR + GitHub Actions
```

---

## Screenshots

Add screenshots here before deleting the EC2 instance.

Recommended screenshots:

```txt
1. GitHub repository homepage
2. Project folder structure
3. Pinecone index
4. GitHub Actions successful workflow
5. AWS ECR image
6. EC2 running instance
7. Docker container running on EC2
8. Deployed chatbot in browser
9. Chatbot answer with source citations
```

Example:

```md
### Chatbot UI

![Chatbot UI](screenshots/chatbot-ui.png)

### Source Citations

![Source Citations](screenshots/source-citations.png)

### GitHub Actions Deployment

![GitHub Actions](screenshots/github-actions.png)
```

---

## Completed Work

- Created project repository
- Built notebook-based RAG pipeline
- Loaded and chunked PDF documents
- Added metadata for source citations
- Created HuggingFace embeddings
- Stored vectors in Pinecone
- Connected Gemini LLM
- Built Flask chatbot
- Improved frontend UI
- Added source citation formatting
- Dockerized the app
- Created GitHub Actions CI/CD workflow
- Deployed on AWS EC2 using Docker and ECR
- Debugged memory issue and redeployed on larger EC2 instance

---

## Roadmap

### Phase 1 — Current AWS Version

Status: Completed

```txt
Medical RAG chatbot
Gemini API
Pinecone vector database
Flask UI
Docker
AWS EC2 deployment
GitHub Actions CI/CD
```

### Phase 2 — Domain Switch Option

Status: Planned

The app is designed to support multiple domains.

Planned domains:

```txt
medical
machine_learning
llm
```

Current limitation:

```python
DOMAIN = "medical"
```

Planned improvements:

- Add PDFs into separate folders:
  - `data/medical/`
  - `data/machine_learning/`
  - `data/llm/`
- Update `store_index.py` to accept a domain argument
- Upload each domain with correct metadata
- Use one Pinecone index with metadata filtering
- Connect frontend dropdown to backend retriever filter
- Test retrieval separately for each domain

Future command style:

```bash
python store_index.py --domain medical
python store_index.py --domain machine_learning
python store_index.py --domain llm
```

### Phase 3 — Google Cloud Deployment

Status: Planned

Deploy the Dockerized app to Google Cloud Run using GCP credits.

Planned services:

```txt
Google Cloud Run
Google Artifact Registry
Google Secret Manager
Vertex AI
```

Planned steps:

- Enable Cloud Run API
- Enable Artifact Registry API
- Enable Vertex AI API
- Push Docker image to Artifact Registry
- Deploy container to Cloud Run
- Add secrets/environment variables
- Test public Cloud Run URL

### Phase 4 — Vertex AI Gemini

Status: Planned

Current version uses Gemini API key.

Future version will use Vertex AI Gemini with Google Cloud authentication.

Planned change:

```python
from langchain_google_vertexai import ChatVertexAI
```

Future model configuration:

```python
chat_model = ChatVertexAI(
    model="gemini-2.5-flash",
    project=GOOGLE_CLOUD_PROJECT,
    location=GOOGLE_CLOUD_LOCATION,
    temperature=0.2
)
```

### Phase 5 — Redis Conversation Memory

Status: Planned

Current memory is temporary/in-memory.

Future improvement:

```txt
Redis-backed LangChain conversation memory
```

Planned features:

- Store chat history by session ID
- Persist memory outside Flask process
- Add TTL for old sessions
- Clear memory from Clear Chat button
- Support multi-user deployed memory

Example future `.env`:

```env
REDIS_URL=redis://localhost:6379/0
CHAT_SESSION_TTL=3600
```

### Phase 6 — Evaluation System

Status: Planned

Add a test set to evaluate answer quality and grounding.

Planned files:

```txt
evaluation/questions.json
evaluation/evaluate.py
evaluation/results.csv
```

Planned metrics:

- Number of test questions
- Answers with citations
- Keyword match score
- Unknown answer count
- Retrieved chunks per question

Example result table:

| Metric                     | Result |
| -------------------------- | ------ |
| Test Questions             | 30     |
| Answers with Citations     | 30/30  |
| Average Keyword Match      | 80%+   |
| Retrieved Chunks per Query | 5      |

---

## Future Improvements

- Add full multi-domain switching
- Add machine learning and LLM document domains
- Deploy to Google Cloud Run
- Use Vertex AI Gemini instead of API-key Gemini
- Add Redis-backed persistent conversation memory
- Add evaluation pipeline
- Add admin PDF upload panel
- Add FAISS/Chroma local vector database option
- Add user authentication
- Add Nginx and HTTPS
- Deploy on ECS/Fargate or Cloud Run for production-style deployment
- Add monitoring and logging

---

## Limitations

- The current app supports only the medical domain in production.
- Domain switching is partially prepared but not fully implemented yet.
- Conversation memory is planned for improvement with Redis.
- PDF documents are not included in the repository due to copyright and size concerns.
- The app depends on Pinecone availability.
- A low-memory cloud instance may fail due to embedding model memory usage.
- Medical answers are educational only and should not be treated as professional medical advice.

---

## Resume Bullet Points

- Built an end-to-end source-grounded RAG chatbot using LangChain, Gemini, Pinecone, Flask, and HuggingFace embeddings to answer questions from domain-specific PDF documents.
- Designed a metadata-based citation system with source, page, domain, and chunk IDs to improve answer traceability and reduce hallucination risk.
- Dockerized the application and deployed it to AWS EC2 using AWS ECR and GitHub Actions CI/CD with a self-hosted runner.
- Improved the tutorial-based implementation by adding Gemini support, source citations, domain-ready architecture, a custom UI, and planned Redis-based conversation memory.
- Debugged cloud deployment issues including Docker build failures, `.dockerignore` conflicts, and EC2 memory-related container exits.

---

## Author

Kunal Pandey

---

## License

This project is for educational and portfolio purposes.
