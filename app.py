import os
import requests
import fitz  # PyMuPDF
import uvicorn
import logging
from typing import List, Optional, Set

from fastapi import FastAPI, HTTPException, Depends, Header, APIRouter
from pydantic import BaseModel, HttpUrl, Field
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_groq import ChatGroq  # <-- CHANGE: Import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser # <-- CHANGE: Import Pydantic parser

# --- 1. INITIAL SETUP & CONFIGURATION ---

# Load environment variables from a .env file
load_dotenv()

# Set up basic logging to monitor the server's activity
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- CONFIGURATION & CONSTANTS ---
# Fetch API keys and other configuration from environment variables

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "new-bajaj")
EXPECTED_BEARER_TOKEN = os.getenv("BEARER_TOKEN", "af8fd52f8162092c73668c2ecfd18a703b6393c5488e994435f391f7257f708c")

# Configuration for the embedding model and Pinecone
VECTOR_DIMENSION = 384
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

# <-- CHANGE: Validate Groq and Pinecone keys
if not PINECONE_API_KEY or not GROQ_API_KEY:
    raise ValueError("FATAL: PINECONE_API_KEY and GROQ_API_KEY must be set in the .env file.")

# --- 2. PYDANTIC MODELS FOR DATA VALIDATION ---

class QueryRequest(BaseModel):
    documents: HttpUrl
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

class AnswerList(BaseModel):
    answers: List[str] = Field(description="A list of answers to the user's questions.")


# --- 3. INITIALIZE EXTERNAL SERVICES ---

try:
    logging.info("Initializing external services...")
    embeddings_model = FastEmbedEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # <-- CHANGE: Initialize Groq LLM instead of Gemini
    # We use a specific model like 'llama3-8b-8192' for speed and capability.
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama3-8b-8192",
        temperature=0
    )

    # <-- CHANGE: Initialize a Pydantic Output Parser
    # This parser will instruct the LLM on how to format the output and parse it into our `AnswerList` model.
    parser = PydanticOutputParser(pydantic_object=AnswerList)

    logging.info("Services initialized successfully.")
except Exception as e:
    logging.error(f"Failed to initialize services: {e}", exc_info=True)
    raise

# --- 4. FASTAPI APPLICATION SETUP ---

app = FastAPI(
    title="Optimized RAG API with Groq", # <-- CHANGE: Updated title
    description="An API to answer questions about a document using a Retrieval-Augmented Generation pipeline with Groq."
)
router = APIRouter(prefix="/api/v1")

# --- 5. AUTHENTICATION DEPENDENCY ---

async def verify_token(authorization: Optional[str] = Header(None)):
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header is missing.")
    try:
        auth_type, token = authorization.split()
        if auth_type.lower() != "bearer" or token != EXPECTED_BEARER_TOKEN:
            raise HTTPException(status_code=403, detail="Invalid authentication credentials.")
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid authorization header format. Must be 'Bearer <token>'.")
    return token

# --- 6. SERVER STARTUP EVENT ---

@app.on_event("startup")
async def startup_event():
    logging.info("--- Server Starting Up: Checking Pinecone Index ---")
    try:
        if PINECONE_INDEX_NAME not in pc.list_indexes().names():
            logging.warning(f"Pinecone index '{PINECONE_INDEX_NAME}' not found. Creating it now...")
            pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=VECTOR_DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION)
            )
            logging.info(f"Successfully created index: '{PINECONE_INDEX_NAME}'")
        else:
            index_description = pc.describe_index(PINECONE_INDEX_NAME)
            if index_description.dimension != VECTOR_DIMENSION:
                logging.error(f"CRITICAL: Index dimension mismatch. Expected {VECTOR_DIMENSION}, found {index_description.dimension}.")
                raise ValueError("Mismatched Pinecone index dimension. Please delete the index or update the script.")
            logging.info(f"Target Pinecone index '{PINECONE_INDEX_NAME}' already exists with correct dimension.")
    except Exception as e:
        logging.error(f"Could not connect to or configure Pinecone index: {e}", exc_info=True)
        raise

# --- 7. API ENDPOINT DEFINITION ---

@router.post("/hackrx/run", response_model=QueryResponse, tags=["RAG Pipeline"])
async def run_rag_pipeline(request: QueryRequest, token: str = Depends(verify_token)):
    try:
        # Step 1 & 2: Download and Extract Text from PDF
        logging.info(f"Step 1: Downloading document from {request.documents}")
        response = requests.get(str(request.documents))
        response.raise_for_status()
        file_content = response.content
        source_filename = os.path.basename(str(request.documents).split('?')[0])

        logging.info("Step 2: Extracting text from PDF...")
        docs = []
        with fitz.open(stream=file_content, filetype="pdf") as doc:
            for page_num, page in enumerate(doc):
                text = page.get_text(sort=True)
                if text.strip():
                    docs.append(Document(page_content=text, metadata={"source": source_filename, "page": page_num + 1}))
        if not docs:
            raise HTTPException(status_code=400, detail="The provided document has no extractable text.")
        logging.info(f"Extracted text from {len(docs)} pages.")

        # Step 3: Split document into chunks
        logging.info("Step 3: Splitting document into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        doc_chunks = text_splitter.split_documents(docs)
        logging.info(f"Split document into {len(doc_chunks)} chunks.")

        # Step 4: Embed chunks and upsert to Pinecone
        logging.info(f"Step 4: Embedding chunks and upserting to '{PINECONE_INDEX_NAME}'...")
        vectorstore = PineconeVectorStore.from_documents(
            documents=doc_chunks,
            embedding=embeddings_model,
            index_name=PINECONE_INDEX_NAME
        )
        logging.info("Upsert to Pinecone complete.")

        # --- Step 5: More Efficient Context Retrieval ---
        logging.info(f"Step 5: Retrieving relevant context for all {len(request.questions)} questions...")
        
        unique_contents: Set[str] = set()
        consolidated_context: List[Document] = []
        for question in request.questions:
            retrieved_docs = vectorstore.similarity_search(question, k=2)
            for doc in retrieved_docs:
                if doc.page_content not in unique_contents:
                    unique_contents.add(doc.page_content)
                    consolidated_context.append(doc)
        
        logging.info(f"Consolidated {len(consolidated_context)} unique document chunks for context.")

        # --- Step 6: Answer all questions in a single, batched API call ---
        logging.info("Step 6: Answering all questions with a single Groq call...") # <-- CHANGE
        all_questions_str = "\n".join(f"- {q}" for q in request.questions)
        
        # <-- CHANGE: Update prompt to include format instructions from the parser
        prompt_template = PromptTemplate(
            template="""
            Based *only* on the following context, please provide a direct and concise answer for each of the questions listed below.
            You must provide an answer for every question.

            CONTEXT:
            {context}

            QUESTIONS:
            {questions}

            {format_instructions}
            """,
            input_variables=["context", "questions"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )

        # <-- CHANGE: Chain now includes the parser at the end
        chain = prompt_template | llm | parser

        logging.info("Invoking LLM for batched response...")
        # The parser will automatically process the LLM's string output into an `AnswerList` object.
        result = chain.invoke({
            "context": "\n\n".join(doc.page_content for doc in consolidated_context),
            "questions": all_questions_str
        })

        final_answers = result.answers

        logging.info("--- RAG Pipeline Completed Successfully ---")
        return QueryResponse(answers=final_answers)

    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to download document: {e}")
        raise HTTPException(status_code=400, detail=f"Could not fetch document from URL: {e}")
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logging.error(f"An unexpected error occurred during the RAG process: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected server error occurred. Please check logs.")

app.include_router(router)