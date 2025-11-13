import streamlit as st
from pydantic_ai import Agent, ModelRetry, DeferredToolRequests, DeferredToolResults, ToolDenied
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.models.google import GoogleModel
from subprocess import run
from dotenv import load_dotenv
from os import getenv
import os
import chromadb
import requests  # For Docker Hub search
import json      # For Docker Hub search
import logfire
import asyncio
import threading
from typing import Any, Coroutine
import traceback  # For detailed error logging

# Load environment variables
load_dotenv()


DB_PATH = "./chroma_docker_db"
COLLECTION_NAME = "docker_help"


logfire.configure(token=getenv("LOGFIRE_WRITE_TOKEN"))
logfire.instrument_pydantic_ai()


class EventLoopThread:
    
    def __init__(self):
        self.loop = None
        self.thread = None
        self._started = False
        
    def start(self):
        if self._started:
            return
            
        def run_loop():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.loop.run_forever()
        
        self.thread = threading.Thread(target=run_loop, daemon=True)
        self.thread.start()
        
        while self.loop is None:
            threading.Event().wait(0.01)
        
        self._started = True
    
    def run_coroutine(self, coro: Coroutine) -> Any:
        
        if not self._started:
            self.start()
        
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        return future.result()
    
    def stop(self):
        
        if self.loop and self._started:
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.thread.join(timeout=5)
            self._started = False


if "event_loop_thread" not in st.session_state:
    st.session_state.event_loop_thread = EventLoopThread()
    st.session_state.event_loop_thread.start()


def run_async(coro: Coroutine) -> Any:
    
    return st.session_state.event_loop_thread.run_coroutine(coro)


async def validate_google_api_key(api_key: str) -> tuple[bool, str]:
    
    try:
        test_model = GoogleModel(
            'gemini-2.5-flash',
            provider=GoogleProvider(api_key=api_key)
        )
        
        test_agent = Agent(test_model)
        
        result = await test_agent.run("Hello")
        
        return True, "API key is valid"
    except Exception as e:
        error_msg = str(e)
        if "API_KEY_INVALID" in error_msg or "invalid" in error_msg.lower():
            return False, "Invalid API key"
        elif "quota" in error_msg.lower():
            return False, "API key quota exceeded"
        elif "permission" in error_msg.lower():
            return False, "API key lacks required permissions"
        else:
            return False, f"Validation failed: {error_msg}"



def gather_context(command: str) -> str:
    """
    This tool MUST ONLY be used to gather context from the user's Docker environment.
    ONLY call this tool for contextual questions.
    It must never execute final commands.
    This tool will execute an arbitrary command on the user machine's CLI.
    
    Args:
        command: The CLI command to gather required context from the user's machine.
    """
    result = run(command, shell=True, capture_output=True, text=True)
    if result.stderr:
        error_msg = f"There was an error in executing the command: {result.stderr}"
        print(f"--- Tool Error [gather_context] ---\n{error_msg}\n---------------------------------")
        raise ModelRetry(error_msg)
    
    output = result.stdout
    print(f"--- Tool Output [gather_context] ---\n{output}\n---------------------------------")
    return output


def get_chroma_collection():
    
    if not os.path.exists(DB_PATH):
        return None
    try:
        client = chromadb.PersistentClient(path=DB_PATH)
        collection = client.get_collection(COLLECTION_NAME)
        return collection
    except Exception as e:
        print(f"--- ERROR loading ChromaDB in get_chroma_collection ---")
        traceback.print_exc()
        return None

def search_docker_documentation(query: str) -> str:
    """
    Contains all docker commands help outputs.
    Searches the *local* Docker help documentation (ChromaDB) to find
    relevant help texts, command usages, and flag descriptions.
    Use this to answer questions about *how* to use Docker commands.
    
    Args:
        query: The natural language query to search for.
    """
    collection = get_chroma_collection()
    if collection is None:
        error_msg = f"Error: ChromaDB not found at '{DB_PATH}' or failed to load. Please run the `create_docker_db.py` script first to create the documentation database."
        print(f"--- Tool Error [search_docker_documentation] ---\n{error_msg}\n---------------------------------")
        return error_msg
    
    try:
        results = collection.query(query_texts=[query], n_results=3)
        
        if not results['documents'] or not results['documents'][0]:
            output = "No relevant documentation found for that query."
            print(f"--- Tool Output [search_docker_documentation] ---\n{output}\n---------------------------------")
            return output
        
        context = "--- Relevant Docker Documentation ---\n"
        for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
            context += f"\n**Command:** {meta.get('command')}\n"
            context_type = meta.get('type', 'general')
            context += f"**Section:** {context_type}\n"
            if meta.get('flag'):
                context += f"**Flag:** {meta.get('flag')}\n"
            context += f"**Documentation:** {doc}\n"
            context += "------------------------------------\n"
        
        print(f"--- Tool Output [search_docker_documentation] ---\n{context}\n---------------------------------")
        return context
    except Exception as e:
        print("--- ERROR IN search_docker_documentation ---")
        traceback.print_exc()
        error_msg = f"Error querying ChromaDB: {e}"
        print(f"--- Tool Error [search_docker_documentation] ---\n{error_msg}\n---------------------------------")
        return error_msg


def get_dockerhub_token() -> str | None:
    
    
    if "dockerhub_jwt_token" in st.session_state:
        return st.session_state.dockerhub_jwt_token
    
    
    username = st.session_state.get("dockerhub_username")
    pat = st.session_state.get("dockerhub_pat")
    
    if not username or not pat:
        print("No Docker Hub credentials provided. Proceeding with anonymous search.")
        return None
        
    # Fetch new token
    print("Fetching new Docker Hub JWT token...")
    try:
        login_url = "https://hub.docker.com/v2/users/login/"
        login_data = {"username": username, "password": pat}
        response = requests.post(login_url, json=login_data, timeout=5)
        response.raise_for_status()
        
        token = response.json().get("token")
        if not token:
            print("Login failed: 'token' not found in response.")
            return None
            
        
        st.session_state.dockerhub_jwt_token = token
        return token
        
    except requests.exceptions.RequestException as e:
        print(f"Error authenticating with Docker Hub: {e}")
        st.error(f"Docker Hub login failed: {e}")
        return None

def search_docker_hub(image_query: str) -> str:
    """
    Searches Docker Hub v2 API for images.
    Uses authentication if credentials (username + PAT) are provided
    in the sidebar.
    
    Args:
        image_query: The name of the software to search for (e.g., "mongo", "node").
    """
    print(f"Searching Docker Hub (v2 API) for: {image_query}")
    headers = {"User-Agent": "DockerGPT-Streamlit-App"}
    
    
    jwt_token = get_dockerhub_token()
    if jwt_token:
        headers["Authorization"] = f"JWT {jwt_token}"
        print("Performing authenticated search.")
    else:
        print("Performing anonymous search.")

    try:
        
        search_url = "https://hub.docker.com/v2/search/repositories/"
        params = {'query': image_query, 'page_size': 5}
        
        response = requests.get(search_url, params=params, headers=headers, timeout=5)
        response.raise_for_status()
        
        data = response.json()
        search_results = data.get('results', [])
        
        if not search_results:
            output = f"No Docker Hub images found for query: {image_query}"
            print(f"--- Tool Output [search_docker_hub] ---\n{output}\n---------------------------------")
            return output

        
        results = "--- Docker Hub Search Results ---\n"
        for i, item in enumerate(search_results[:3]): 
            name = item.get('repo_name')
            description = item.get('short_description', 'No description.')
            stars = item.get('star_count', 0)
            
            
            if "/" not in name and "library" not in name:
                name = f"library/{name}"

            results += f"{i+1}. Image: {name} (Stars: {stars})\n   Description: {description}\n"
        
        print(f"--- Tool Output [search_docker_hub] ---\n{results}\n---------------------------------")
        return results

    except requests.exceptions.RequestException as e:
        print(f"Docker Hub search failed: {e}")
        error_msg = f"Error connecting to Docker Hub: {e}"
        print(f"--- Tool Error [search_docker_hub] ---\n{error_msg}\n---------------------------------")
        if "401" in str(e) and "dockerhub_jwt_token" in st.session_state:
            del st.session_state.dockerhub_jwt_token
            error_msg += " (Token was invalid, cleared cache. Please check credentials.)"
        return error_msg
    except Exception as e:
        print(f"Error parsing Docker Hub response: {e}")
        error_msg = f"Error parsing Docker Hub response: {e}"
        print(f"--- Tool Error [search_docker_hub] ---\n{error_msg}\n---------------------------------")
        return error_msg


def list_directory(path: str = '.') -> str:
    """
    Lists the contents of a directory (files and subdirectories).
    This helps understand the project structure.
    Only lists up to 2 levels deep.
    Requires user approval.
    
    Args:
        path: The directory path to list. Defaults to current directory.
    """
    print(f"Listing directory: {path}")
    try:
        output = f"Directory listing for: {path}\n"
        for root, dirs, files in os.walk(path, topdown=True):
            level = root.replace(path, '').count(os.sep)
            if level > 1: 
                dirs[:] = [] 
                files[:] = []
                continue
                
            indent = '  ' * level
            output += f"{indent}[{os.path.basename(root)}/]\n"
            file_indent = '  ' * (level + 1)
            for f in files:
                output += f"{file_indent}{f}\n"
        
        output_str = output.strip()
        print(f"--- Tool Output [list_directory] ---\n{output_str}\n---------------------------------")
        return output_str
    except FileNotFoundError:
        error_msg = f"Error: Directory not found at {path}"
        print(f"--- Tool Error [list_directory] ---\n{error_msg}\n---------------------------------")
        return error_msg
    except Exception as e:
        error_msg = f"Error listing directory: {e}"
        print(f"--- Tool Error [list_directory] ---\n{error_msg}\n---------------------------------")
        return error_msg

def read_file(filepath: str) -> str:
    """
    Reads the content of a specified file.
    Use this to read project files like package.json, requirements.txt, etc.
    Reads a maximum of 2000 characters.
    Requires user approval.
    
    Args:
        filepath: The path to the file.
    """
    print(f"Reading file: {filepath}")
    try:
        if ".." in filepath:
            error_msg = "Error: File path cannot contain '..'"
            print(f"--- Tool Error [read_file] ---\n{error_msg}\n---------------------------------")
            return error_msg
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read(2000) 
        
        if len(content) == 2000:
            output = f"--- Content of {filepath} (first 2000 chars) ---\n{content}\n... (file truncated)"
        else:
            output = f"--- Content of {filepath} ---\n{content}"
        
        print(f"--- Tool Output [read_file] ---\n{output}\n---------------------------------")
        return output
    except FileNotFoundError:
        error_msg = f"Error: File not found at {filepath}"
        print(f"--- Tool Error [read_file] ---\n{error_msg}\n---------------------------------")
        return error_msg
    except Exception as e:
        error_msg = f"Error reading file: {e}"
        print(f"--- Tool Error [read_file] ---\n{error_msg}\n---------------------------------")
        return error_msg




def initialize_agent(model_choice, api_key=None):
    """Initialize the agent based on model choice."""
    if model_choice == "Local (Ollama)":
        if not os.path.exists(DB_PATH):
            raise FileNotFoundError(f"ChromaDB not found at '{DB_PATH}'. Please run `create_docker_db.py` first.")
        
        ollama_model = OpenAIChatModel(
            model_name='qwen3:1.7b',
            provider=OllamaProvider(base_url='http://localhost:11434/v1'),
        )
        agent = Agent(
            ollama_model,
            system_prompt=[
                "You are DockerGPT — an expert Docker assistant that uses tools to gather accurate information.",
                "Your tools:",
                "- search_docker_documentation(query) → retrieves Docker help, flags, and usage text from the local ChromaDB.",
                "- list_directory(path) → lists files in a project directory.",
                "- read_file(filepath) → reads project files such as package.json and requirements.txt.",
                "- search_docker_hub(name) → finds Docker Hub images.",
                "- gather_context(cmd) → runs system commands, ONLY for live container or environment information.",

                "==============================",
                "## CORE BEHAVIOR",
                "==============================",

                "### 1. DOCKER DOCUMENTATION WORKFLOW (MANDATORY)",
                "For ANY of the following:",
                "- questions about Docker commands",
                "- questions about Docker flags",
                "- questions about how a command works",
                "- 'what does this do?' questions",
                "- usage, syntax, options, or help text",
                "- comparing commands or flags",
                "You MUST ALWAYS call search_docker_documentation(query) FIRST.",
                "Never answer documentation questions from your own knowledge.",
                "Never guess Docker help text or flag definitions.",
                "After receiving tool output, summarize the correct answer based on it.",
                "For documentation questions, your final answer is a normal explanation, NOT a command or YAML.",

                "### 2. DOCKER-COMPOSE WORKFLOW (STRICT STEPS)",
                "Use this ONLY when the user explicitly asks for a docker-compose.yaml or containerizing a project.",
                "Follow this order:",
                "A. Call list_directory(path) to inspect the project structure IF user did not provide file contents.",
                "B. Call read_file(filepath) for all relevant files: package.json, requirements.txt, Dockerfile, README.md, pyproject.toml, etc.",
                "C. Call search_docker_hub(library) to confirm correct image names for dependencies.",
                "D. FINALLY provide ONLY the docker-compose.yaml with no explanation, comments, or prose.",

                "### 3. SIMPLE COMMAND GENERATION WORKFLOW",
                "If the user wants a specific Docker command:",
                "- Gather missing details using tools (search_docker_hub, gather_context, etc.) only if needed.",
                "- Then output EXACTLY ONE Docker command, nothing else.",
                "No explanation. No extra text. Just the command.",

                "### 4. gather_context RULES",
                "You may ONLY call gather_context(cmd) when:",
                "- The user explicitly asks for live information (running containers, networks, images, volumes, logs, etc.).",
                "- Or when generating a command requires real system data.",
                "NEVER call gather_context for documentation or flag questions.",
                "NEVER call gather_context to explore the system unless the user asks.",

                "### 5. SAFETY & ACCURACY RULES",
                "Never guess image names — always use search_docker_hub when unsure.",
                "Never guess file contents — always use read_file when needed.",
                "Never hallucinate Docker help text — always use search_docker_documentation for ANY command/flag question.",
                "Your output depends strictly on user intent:",
                "- Docs question → normal explanation (after RAG tool).",
                "- Docker command → output a single command.",
                "- docker-compose.yaml → output ONLY the YAML.",

                "### 6. DECISION CHECKLIST (ALWAYS RUN INTERNALLY)",
                "1. Is the user asking about a Docker command or flag?",
                "   → ALWAYS call search_docker_documentation first.",
                "2. Is the user asking for a docker-compose file?",
                "   → Run the strict compose workflow (list → read → hub → output YAML).",
                "3. Does the user want a runnable Docker command?",
                "   → Gather info only if required, then output exactly one command.",
                "4. Did the user already provide file contents?",
                "   → DO NOT call list_directory or read_file for them.",
                "5. Does the user want live system/container info?",
                "   → Use gather_context.",

                "### 7. OUTPUT TYPES",
                "Your answer must be exactly one of:",
                "- A normal explanation (for documentation questions)",
                "- A single docker command",
                "- A docker-compose.yaml",
                "- A DeferredToolRequest if you need a tool",
                "Never combine explanation + command/YAML.",
                "Never add commentary outside these formats.",

                "Follow these rules precisely."
            ],
            output_type=[str, DeferredToolRequests]
        )
        # Add all tools for local agent
        agent.tool_plain(requires_approval=False)(search_docker_documentation)
        agent.tool_plain(requires_approval=False)(search_docker_hub)
        agent.tool_plain(requires_approval=True)(list_directory)
        agent.tool_plain(requires_approval=True)(read_file)

    else:
        if not api_key:
            raise ValueError("Google API Key is required for remote model")
        
        model = GoogleModel(
            'gemini-2.5-flash',
            provider=GoogleProvider(api_key=api_key)
        )
        agent = Agent(
            model,

            
            system_prompt=[
                'You are an expert Docker assistant. Your goal is to provide a single, final Docker command OR a complete `docker-compose.yaml` file.',
                'You MUST gather all necessary information *before* providing the final answer. Do not guess or hallucinate file contents or image names.',
                
                '--- REASONING STEPS (FOLLOW THIS ORDER) ---',
                
                '1. **Analyze the User Prompt FIRST.**',
                '   - If the user provides the *content* of a file (like `package.json`), use that information directly. **DO NOT** use `read_file` or `list_directory` if the information is already in the prompt.',
                '   - If the user provides a *directory path*, your first step is the `docker-compose.yaml` workflow below.',
                '   - If the user asks a *general question* (e.g., "how to use -v"), use the `Simple Command Workflow` below.',

                '2. **`docker-compose.yaml` Workflow (This is a strict multi-step process):**',
                '   - **A.** If (and only if) the user asked for a compose file for a project/directory and did NOT provide file contents, your *first* action MUST be `list_directory(...)` to see the project structure.',
                '   - **B.** **CRITICAL STEP:** After you get the directory list, your *second* action MUST be to use `read_file(...)` on any relevant files you find (e.g., `README.md`, `package.json`, `requirements.txt`). Read all of them if they exist.',
                '   - **C.** After you get the file *content*, your *third* action is to use `search_docker_hub(...)` to find the correct image names for the libraries/dependencies you found (e.g., "node", "redis", "postgres").',
                '   - **D.** **Only after** you have completed steps A, B, and C, you can generate the final `docker-compose.yaml`.',

                '3. **`Simple Command` Workflow:**',
                '   - If the user just needs an *image name*, use `search_docker_hub(...)` and answer.',
                '   - If live data (like container IDs) is needed, use `gather_context`.',

                '4. **Final Answer:**',
                '   - The final output must ONLY be the command string or the YAML content.',
                '   - Do not guess. If you lack information (e.g., `read_file` failed), ask the user for it.'
            ],
            

            output_type=[str, DeferredToolRequests]
        )
        
        agent.tool_plain(requires_approval=False)(search_docker_hub)
        agent.tool_plain(requires_approval=True)(list_directory)
        agent.tool_plain(requires_approval=True)(read_file)

    
    
    agent.tool_plain(requires_approval=True)(gather_context)
    return agent


st.set_page_config(page_title="DockerGPT", page_icon="🐳", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent" not in st.session_state:
    st.session_state.agent = None
if "pending_approval" not in st.session_state:
    st.session_state.pending_approval = None
if "result" not in st.session_state:
    st.session_state.result = None
if "model_selected" not in st.session_state:
    st.session_state.model_selected = False
if "google_api_key" not in st.session_state:
    st.session_state.google_api_key = getenv("GOOGLE_API_KEY", "")
if "model_choice" not in st.session_state:
    st.session_state.model_choice = None
if "show_api_key_input" not in st.session_state:
    st.session_state.show_api_key_input = False
if "last_approved_command" not in st.session_state:
    st.session_state.last_approved_command = None
if "api_key_validated" not in st.session_state:
    st.session_state.api_key_validated = False
if "api_key_invalid" not in st.session_state:
    st.session_state.api_key_invalid = False
if "agent_message_history" not in st.session_state:
    st.session_state.agent_message_history = []
if "dockerhub_username" not in st.session_state:
    st.session_state.dockerhub_username = ""
if "dockerhub_pat" not in st.session_state:
    st.session_state.dockerhub_pat = ""


with st.sidebar:
    st.title(" DockerGPT")
    st.markdown("---")
    
    
    if not st.session_state.model_selected:
        st.subheader("Model Selection")
        model_choice = st.radio(
            "Choose your model:",
            ["Local (Ollama)", "Remote (Google Gemini)"],
            help="Local model uses Ollama with qwen3:1.7b. Remote uses Google Gemini 2.5 Flash."
        )
        
        st.session_state.model_choice = model_choice
        st.markdown("---")
        
        if model_choice == "Local (Ollama)":
            st.info("###  Ollama Setup Instructions")
            st.markdown("""
            **Installation:**
            1. Visit [ollama.ai](https://ollama.ai) to download Ollama
            2. Install and start Ollama
            3. Run: `ollama pull qwen3:1.7b`
            
            ** Disclaimer:**
            The qwen3:1.7b model (~1GB) will be downloaded automatically if not present. Ensure you have sufficient disk space and a stable internet connection.
            """)

            st.markdown("---")
            
            st.markdown("RAG Database Setup")
            if os.path.exists(DB_PATH):
                st.success(f" Documentation DB found at `{DB_PATH}`.")
            else:
                st.error(f" Documentation DB not found.")
                st.warning(f"Please run `python create_docker_db.py` in your terminal to build the local help database before initializing the agent.")
        
        elif model_choice == "Remote (Google Gemini)":
            st.markdown("### API Key Configuration")
            
            if st.session_state.api_key_invalid:
                st.error(" Your current API key is invalid. Please enter a valid API key to continue.")
                st.session_state.google_api_key = ""
                st.session_state.api_key_validated = False
                st.session_state.api_key_invalid = False
            
            if not st.session_state.google_api_key or not st.session_state.api_key_validated:
                api_key_input = st.text_input(
                    "Google API Key",
                    type="password",
                    placeholder="Enter your Google API Key",
                    help="Required for using Google Gemini model"
                )
                
                if api_key_input:
                    if st.button("Validate API Key", type="secondary"):
                        with st.spinner("Validating API key..."):
                            try:
                                is_valid, message = run_async(validate_google_api_key(api_key_input))
                                
                                if is_valid:
                                    st.session_state.google_api_key = api_key_input
                                    st.session_state.api_key_validated = True
                                    st.success(f" {message}")
                                else:
                                    st.error(f" {message}")
                                    st.session_state.api_key_validated = False
                            except Exception as e:
                                st.error(f" Validation error: {str(e)}")
                                st.session_state.api_key_validated = False
            else:
                st.success(" API Key validated")
                if st.button("Update API Key"):
                    st.session_state.show_api_key_input = True
                
                if st.session_state.show_api_key_input:
                    new_api_key = st.text_input(
                        "New Google API Key",
                        type="password",
                        placeholder="Enter new Google API Key"
                    )
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Validate & Save"):
                            if new_api_key:
                                with st.spinner("Validating new API key..."):
                                    try:
                                        is_valid, message = run_async(validate_google_api_key(new_api_key))
                                        
                                        if is_valid:
                                            st.session_state.google_api_key = new_api_key
                                            st.session_state.api_key_validated = True
                                            st.session_state.show_api_key_input = False
                                            st.success("API Key validated and updated!")
                                            st.rerun()
                                        else:
                                            st.error(f" {message}")
                                    except Exception as e:
                                        st.error(f" Validation error: {str(e)}")
                            else:
                                st.error("Please enter a valid API key")
                    with col2:
                        if st.button("Cancel"):
                            st.session_state.show_api_key_input = False
                            st.rerun()
        
        can_initialize = (
            (model_choice == "Local (Ollama)" and os.path.exists(DB_PATH)) or 
            (model_choice == "Remote (Google Gemini)" and st.session_state.api_key_validated)
        )
        
        if can_initialize:
            if st.button("Initialize Agent", type="primary"):
                with st.spinner("Initializing agent..."):
                    try:
                        st.session_state.agent = initialize_agent(
                            model_choice, 
                            st.session_state.google_api_key if model_choice == "Remote (Google Gemini)" else None
                        )
                        st.session_state.model_selected = True
                        st.success(f"Agent initialized with {model_choice}!")
                        st.rerun()
                    except Exception as e:
                        error_msg = str(e)
                        if "API_KEY_INVALID" in error_msg or "invalid" in error_msg.lower():
                            st.error(" API key is invalid. Please update your API key.")
                            st.session_state.api_key_invalid = True
                            st.session_state.api_key_validated = False
                            st.rerun()
                        elif "No module named 'requests'" in error_msg:
                            st.error(" Missing dependency. Please run `pip install requests` in your terminal.")
                        else:
                            st.error(f"Failed to initialize agent: {error_msg}")
        
        elif model_choice == "Local (Ollama)":
             st.warning(" Please run `create_docker_db.py` to build the local database.")
        elif model_choice == "Remote (Google Gemini)":
            if not st.session_state.google_api_key:
                st.warning(" Please enter your Google API Key to continue")
            else:
                st.warning(" Please validate your Google API Key to continue")

    else:
        st.success(" Agent Ready")
        
        if st.session_state.model_choice == "Remote (Google Gemini)":
            if st.button("Update API Key"):
                st.session_state.show_api_key_input = True
            
            if st.session_state.show_api_key_input:
                new_api_key = st.text_input(
                    "New Google API Key",
                    type="password",
                    placeholder="Enter new Google API Key"
                )
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Validate & Save", key="save_running"):
                        if new_api_key:
                            with st.spinner("Validating new API key..."):
                                try:
                                    is_valid, message = run_async(validate_google_api_key(new_api_key))
                                    
                                    if is_valid:
                                        st.session_state.google_api_key = new_api_key
                                        st.session_state.api_key_validated = True
                                        st.session_state.show_api_key_input = False
                                        st.info("API Key validated and updated. Reset agent to use the new key.")
                                        st.rerun()
                                    else:
                                        st.error(f" {message}")
                                except Exception as e:
                                    st.error(f" Validation error: {str(e)}")
                        else:
                            st.error("Please enter a valid API key")
                with col2:
                    if st.button("Cancel", key="cancel_running"):
                        st.session_state.show_api_key_input = False
                        st.rerun()
        
        if st.button("Reset Agent"):
            st.session_state.agent = None
            st.session_state.model_selected = False
            st.session_state.messages = []
            st.session_state.pending_approval = None
            st.session_state.result = None
            st.session_state.agent_message_history = []
            if "dockerhub_jwt_token" in st.session_state:
                del st.session_state.dockerhub_jwt_token
            st.session_state.dockerhub_username = ""
            st.session_state.dockerhub_pat = ""
            st.rerun()
    
    st.markdown("---")
    st.markdown("""
    ### About
    This agent helps you generate Docker commands and `docker-compose.yaml` files.
    
    The agent may request permission to read files or gather context from your environment.
    """)

#  Chat interface
st.title("DockerGPT")

if not st.session_state.model_selected:
    st.info(" Please select and initialize a model from the sidebar to begin.")
else:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if st.session_state.pending_approval:
        approval_data = st.session_state.pending_approval
        
        st.warning(" The agent is requesting permission to run tools:")
        
        with st.container(border=True):
            for call in approval_data["requests"].approvals:
                st.markdown(f"**Tool:** `{call.tool_name}`")
                
                if isinstance(call.args, dict):
                    st.code(json.dumps(call.args, indent=2), language="json")
                else:
                    st.code(str(call.args), language="python")

            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button(" Approve All", key=f"approve_all", type="primary", use_container_width=True):
                    for call in approval_data["requests"].approvals:
                        approval_data["approval_results"].approvals[call.tool_call_id] = True
                    
                    st.session_state.pending_approval = None
                    
                    with st.spinner("Processing..."):
                        try:
                            result = run_async(
                                st.session_state.agent.run(
                                    message_history=approval_data["messages"],
                                    deferred_tool_results=approval_data["approval_results"]
                                )
                            )
                            st.session_state.result = result
                            st.session_state.agent_message_history = result.all_messages()
                        except Exception as e:
                            print("--- AN ERROR OCCURRED (POST-APPROVAL) ---")
                            traceback.print_exc() 
                            error_msg = f"**Type:** `{type(e).__name__}`\n\n**Details:** `{str(e)}`\n\n**Repr:** `{repr(e)}`"
                            
                            st.error(f" Error during agent execution: \n{error_msg}")
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": f" I encountered an error: \n{error_msg}"
                            })
                            st.session_state.result = None
                            st.session_state.agent_message_history = []
                    st.rerun()
            
            with col2:
                if st.button(" Deny All", key=f"deny_all", use_container_width=True):
                    for call in approval_data["requests"].approvals:
                        approval_data["approval_results"].approvals[call.tool_call_id] = ToolDenied(
                            "User has denied the request to execute the command(s)."
                        )
                    st.session_state.pending_approval = None
                    
                    with st.spinner("Cancelling..."):
                        try:
                             result = run_async(
                                st.session_state.agent.run(
                                    message_history=approval_data["messages"],
                                    deferred_tool_results=approval_data["approval_results"]
                                )
                            )
                             st.session_state.result = result
                             st.session_state.agent_message_history = result.all_messages()
                        except Exception as e:
                             st.session_state.messages.append({"role": "assistant", "content": "Request cancelled by user."})
                             st.session_state.result = None
                             st.session_state.agent_message_history = []
                    st.rerun()
    
    if st.session_state.result:
        result = st.session_state.result
        
        if isinstance(result.output, str):
            last_message_role = st.session_state.messages[-1]["role"] if st.session_state.messages else "assistant"
            
            for msg in result.all_messages():
                if hasattr(msg, 'role'):
                    if msg.role == "system" and last_message_role != "system":
                        if msg.content and msg.content.strip():
                            st.session_state.messages.append({"role": "system", "content": msg.content})
                            last_message_role = "system"
                    elif msg.role == "user":
                        last_message_role = "user" 
            
            # Add the final assistant output
            st.session_state.messages.append({"role": "assistant", "content": result.output})
            
            with st.expander(" Usage Statistics"):
                usage = result.usage()
                st.json({
                    "requests": usage.requests,
                    "input_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens,
                    "total_tokens": usage.total_tokens,
                    "details": str(usage.details) if usage.details else None
                })
            
            st.session_state.result = None
            st.rerun()
        elif isinstance(result.output, DeferredToolRequests):
            st.session_state.pending_approval = {
                "requests": result.output,
                "messages": result.all_messages(),
                "approval_results": DeferredToolResults()
            }
            
            tool_calls = []
            for call in result.output.approvals:
                tool_calls.append(f"`{call.tool_name}({call.args})`")
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"Thinking... I need to run the following tool(s):\n" + "\n".join(tool_calls)
            })
            
            st.session_state.result = None
            st.rerun()
        else:
            st.error(f" Unexpected output type: {type(result.output)}")
            st.session_state.messages.append({
                "role": "assistant", 
                "content": f" I encountered an unexpected output type: {type(result.output)}"
            })
            st.session_state.result = None
            st.rerun()
    
    if prompt := st.chat_input("Ask for a Docker command or docker-compose file...", disabled=st.session_state.pending_approval is not None):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = run_async(
                        st.session_state.agent.run(
                            user_prompt=prompt,
                            message_history=st.session_state.agent_message_history
                        )
                    )
                    
                    if isinstance(result.output, DeferredToolRequests):
                        st.session_state.pending_approval = {
                            "requests": result.output,
                            "messages": result.all_messages(),
                            "approval_results": DeferredToolResults()
                        }
                        
                        tool_calls = []
                        for call in result.output.approvals:
                            tool_calls.append(f"`{call.tool_name}({call.args})`")
                        
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"Thinking... I need to run the following tool(s):\n" + "\n".join(tool_calls)
                        })
                        
                        st.rerun()
                    else:
                        st.session_state.result = result
                        st.session_state.agent_message_history = result.all_messages()
                        st.rerun()
                except Exception as e:
                    print("--- AN ERROR OCCURRED (CHAT INPUT) ---")
                    traceback.print_exc() 
                    error_msg = f"**Type:** `{type(e).__name__}`\n\n**Details:** `{str(e)}`\n\n**Repr:** `{repr(e)}`"

                    st.error(f" Error during agent execution: \n{error_msg}")
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": f" I encountered an error: \n{error_msg}"
                    })
                    st.session_state.agent_message_history = []
                    st.rerun()