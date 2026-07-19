"""
Configuration file for the Multi-Agent Medical Chatbot

This file contains all the configuration parameters for the project.

If you want to change the LLM and Embedding model:

you can do it by changing all 'llm' and 'embedding_model' variables present in multiple classes below.

Each llm definition has unique temperature value relevant to the specific class. 
"""

import os
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings, ChatOllama

# Load environment variables from .env file
load_dotenv()

class AgentDecisoinConfig:
    def __init__(self):
        self.llm = ChatOllama(
            model = os.getenv("OLLAMA_MODEL", "mistral:7b"),
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            keep_alive = os.getenv("OLLAMA_KEEP_ALIVE", "60m"),
            num_ctx = 512,   # routing only — reads a short query, no long output needed
            temperature = 0.1,
        )

class ConversationConfig:
    def __init__(self):
        self.llm = ChatOllama(
            model = os.getenv("OLLAMA_MODEL", "mistral:7b"),
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            keep_alive = os.getenv("OLLAMA_KEEP_ALIVE", "60m"),
            num_ctx = 1024,  # enough for prompt + recent history + full reply
            temperature = 0.7,
        )

class WebSearchConfig:
    def __init__(self):
        self.llm = ChatOllama(
            model = os.getenv("OLLAMA_MODEL", "mistral:7b"),
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            keep_alive = os.getenv("OLLAMA_KEEP_ALIVE", "60m"),
            num_ctx = 1536,  # needs to fit web search result text
            temperature = 0.3,
        )
        self.context_limit = 6      # include last 6 messages (3 Q&A pairs) in history

class RAGConfig:
    def __init__(self):
        self.vector_db_type = "qdrant"
        self.embedding_dim = int(os.getenv("OLLAMA_EMBEDDING_DIM", "768"))
        self.distance_metric = "Cosine"
        self.use_local = True
        self.vector_local_path = "./data/qdrant_db"
        self.doc_local_path = "./data/docs_db"
        self.parsed_content_dir = "./data/parsed_docs"
        self.url = os.getenv("QDRANT_URL")
        self.api_key = os.getenv("QDRANT_API_KEY")
        self.collection_name = "medical_assistance_rag"
        self.chunk_size = 512
        self.chunk_overlap = 50
        self.embedding_model = OllamaEmbeddings(
            model = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"),
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        )
        self.llm = ChatOllama(
            model = os.getenv("OLLAMA_MODEL", "mistral:7b"),
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            keep_alive = os.getenv("OLLAMA_KEEP_ALIVE", "60m"),
            num_ctx = 1024,  # general RAG LLM calls
            temperature = 0.3,
        )
        self.summarizer_model = ChatOllama(
            model = os.getenv("OLLAMA_MODEL", "mistral:7b"),
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            keep_alive = os.getenv("OLLAMA_KEEP_ALIVE", "60m"),
            num_ctx = 1024,  # reads a passage and summarises it
            temperature = 0.5,
        )
        self.chunker_model = ChatOllama(
            model = os.getenv("OLLAMA_MODEL", "mistral:7b"),
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            keep_alive = os.getenv("OLLAMA_KEEP_ALIVE", "60m"),
            num_ctx = 512,   # short chunking decisions only
            temperature = 0.0,
        )
        self.response_generator_model = ChatOllama(
            model = os.getenv("OLLAMA_MODEL", "mistral:7b"),
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            keep_alive = os.getenv("OLLAMA_KEEP_ALIVE", "60m"),
            num_ctx = 2048,  # must fit retrieved document chunks — keep high
            temperature = 0.3,
        )

        # Mistral is text-only. Keep image summarization disabled unless you switch
        # to a local vision-capable model and update related code.
        self.enable_image_summarization = False
        self.top_k = 5
        self.vector_search_type = 'similarity'  # or 'mmr'

        self.huggingface_token = os.getenv("HUGGINGFACE_TOKEN")

        self.reranker_model = "cross-encoder/ms-marco-TinyBERT-L-6"
        self.reranker_top_k = 3

        self.max_context_length = 8192  # (Change based on your need) # 1024 proved to be too low (retrieved content length > context length = no context added) in formatting context in response_generator code

        self.include_sources = True  # Show links to reference documents and images along with corresponding query response

        # ADJUST ACCORDING TO ASSISTANT'S BEHAVIOUR BASED ON THE DATA INGESTED:
        self.min_retrieval_confidence = 0.40  # The auto routing from RAG agent to WEB_SEARCH agent is dependent on this value

        self.context_limit = 6      # include last 6 messages (3 Q&A pairs) in history

class MedicalCVConfig:
    def __init__(self):
        self.brain_tumor_classification_model_path = "./agents/image_analysis_agent/brain_tumor_agent/models/brain_tumor_classification.pth"
        self.brain_tumor_segmentation_model_path = "./agents/image_analysis_agent/brain_tumor_agent/models/brain_tumor_segmentation.pth"
        self.brain_tumor_segmentation_output_path = "./uploads/brain_tumor_output/segmentation_plot.png"
        self.chest_xray_model_path = "./agents/image_analysis_agent/chest_xray_agent/models/covid_chest_xray_model.pth"
        self.skin_lesion_model_path = "./agents/image_analysis_agent/skin_lesion_agent/models/checkpointN25_.pth.tar"
        self.skin_lesion_segmentation_output_path = "./uploads/skin_lesion_output/segmentation_plot.png"
        self.llm = ChatOllama(
            model = os.getenv("OLLAMA_VISION_MODEL", "gemma3:4b"),
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            keep_alive = os.getenv("OLLAMA_KEEP_ALIVE", "60m"),
            num_ctx = 1024,  # image description tasks don't need a huge context
            temperature = 0.1,
        )

class SpeechConfig:
    def __init__(self):
        self.eleven_labs_api_key = os.getenv("ELEVEN_LABS_API_KEY")  # Replace with your actual key
        self.eleven_labs_voice_id = "21m00Tcm4TlvDq8ikWAM"    # Default voice ID (Rachel)

class ValidationConfig:
    def __init__(self):
        self.require_validation = {
            "CONVERSATION_AGENT": False,
            "RAG_AGENT": False,
            "WEB_SEARCH_AGENT": False,
            "BRAIN_TUMOR_AGENT": True,
            "CHEST_XRAY_AGENT": True,
            "SKIN_LESION_AGENT": True
        }
        self.validation_timeout = 300
        self.default_action = "reject"

class APIConfig:
    def __init__(self):
        self.host = "0.0.0.0"
        self.port = 8000
        self.debug = True
        self.rate_limit = 10
        self.max_image_upload_size = 5  # max upload size in MB

class UIConfig:
    def __init__(self):
        self.theme = "light"
        # self.max_chat_history = 50
        self.enable_speech = True
        self.enable_image_upload = True

class GoogleOAuthConfig:
    def __init__(self):
        self.client_id = os.getenv("GOOGLE_CLIENT_ID", "")
        self.client_secret = os.getenv("GOOGLE_CLIENT_SECRET", "")
        self.redirect_uri = f"{os.getenv('APP_BASE_URL', 'http://localhost:8000')}/auth/google/callback"
        self.userinfo_url = "https://openidconnect.googleapis.com/v1/userinfo"

class JWTConfig:
    def __init__(self):
        self.secret_key = os.getenv("JWT_SECRET_KEY", "changeme")
        self.algorithm = os.getenv("JWT_ALGORITHM", "HS256")
        self.expire_minutes = int(os.getenv("JWT_EXPIRE_MINUTES", "10080"))

class Config:
    def __init__(self):
        self.agent_decision = AgentDecisoinConfig()
        self.conversation = ConversationConfig()
        self.rag = RAGConfig()
        self.medical_cv = MedicalCVConfig()
        self.web_search = WebSearchConfig()
        self.api = APIConfig()
        self.speech = SpeechConfig()
        self.validation = ValidationConfig()
        self.ui = UIConfig()
        self.google_oauth = GoogleOAuthConfig()
        self.jwt = JWTConfig()
        self.eleven_labs_api_key = os.getenv("ELEVEN_LABS_API_KEY")
        self.tavily_api_key = os.getenv("TAVILY_API_KEY")
        self.max_conversation_history = 20  # Include last 20 messsages (10 Q&A pairs) in history