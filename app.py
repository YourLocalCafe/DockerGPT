import streamlit as st
from pydantic_ai import Agent, ModelRetry, DeferredToolRequests, DeferredToolResults, ToolDenied
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.models.google import GoogleModel
from subprocess import run
from dotenv import load_dotenv
from os import getenv
import logfire
import asyncio

# Load environment variables
load_dotenv()

# Configure logfire
logfire.configure(token=getenv("LOGFIRE_WRITE_TOKEN"))
logfire.instrument_pydantic_ai()


# ... other imports

def run_agent_async(async_func, **kwargs):
    """
    Run an asynchronous function in a new event loop.

    Args:
    async_func (coroutine): The asynchronous function to execute.
    **kwargs: Keyword arguments to pass to the asynchronous function.

    Returns:
    The result from the asynchronous function.
    """
    
    loop = None

    try:
        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(async_func(**kwargs))
        return result  # Return the result
    except Exception as e:
        # Close the existing loop if open
        if loop is not None:
            loop.close()

        # Create a new loop for retry
        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(async_func(**kwargs))
        return result  # Return the result
    finally:
        if loop is not None:
            loop.close()

# Page config
st.set_page_config(page_title="DockerGPT", page_icon="🐳", layout="wide")

# Initialize session state
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
    # Try to get from environment first
    st.session_state.google_api_key = getenv("GOOGLE_API_KEY", "")
if "model_choice" not in st.session_state:
    st.session_state.model_choice = None

if "show_api_key_input" not in st.session_state:
    st.session_state.show_api_key_input = False
if "last_approved_command" not in st.session_state:
    st.session_state.last_approved_command = None

def gather_context(command: str) -> str:
    """
    This tool MUST ONLY be used to gather context from the user's Docker environment.
    ONLY call this tool for contextual questions.
    It must never execute final commands.
    
    Args:
        command: The command to gather required context from the user's machine.
    """
    result = run(command, shell=True, capture_output=True, text=True)
    if result.stderr:
        raise ModelRetry(f"There was an error in executing the command: {result.stderr}")
    return result.stdout

def initialize_agent(model_choice, api_key=None):
    """Initialize the agent based on model choice."""
    if model_choice == "Local (Ollama)":
        ollama_model = OpenAIChatModel(
            model_name='qwen3:1.7b',
            provider=OllamaProvider(base_url='http://localhost:11434/v1'),
        )
        agent = Agent(
            ollama_model,
            system_prompt=[
                'The final output given to the user must only be docker command(s)',
                'The final command is to be given to the user directly as a string in the response, you do not have to give the user the output of the command',
                'Use the gather_context tool available to fill in required container IDs and other command specific information if required/possible.'
            ],
            output_type=[str, DeferredToolRequests]
        )
    else:  # Remote (Google)
        if not api_key:
            raise ValueError("Google API Key is required for remote model")
        
        model = GoogleModel(
            'gemini-2.5-flash',
            provider=GoogleProvider(api_key=api_key)
        )
        agent = Agent(
            model,
            system_prompt=[
                'The final output given to the user must only be docker command(s)',
                'The final command is to be given to the user directly as a string in the response, you do not have to give the user the output of the command',
                'Use the gather_context tool available to fill in required container IDs and other command specific information if required/possible.'
            ],
            output_type=[str, DeferredToolRequests]
        )
    
    # Register the tool
    agent.tool_plain(requires_approval=True)(gather_context)
    return agent

# Sidebar for model selection
with st.sidebar:
    st.title("🐳 DockerGPT")
    st.markdown("---")
    
    if not st.session_state.model_selected:
        st.subheader("Model Selection")
        model_choice = st.radio(
            "Choose your model:",
            ["Local (Ollama)", "Remote (Google Gemini)"],
            help="Local model uses Ollama with qwen3:1.7b. Remote uses Google Gemini 2.5 Flash."
        )
        
        # Store model choice in session state
        st.session_state.model_choice = model_choice
        
        # Show Ollama instructions for local model
        if model_choice == "Local (Ollama)":
            st.info("### 📋 Ollama Setup Instructions")
            st.markdown("""
            **Installation:**
            1. Visit [ollama.ai](https://ollama.ai) to download Ollama
            2. Install and start Ollama
            3. Run: `ollama pull qwen3:1.7b`
            
            **⚠️ Disclaimer:**
            The qwen3:1.7b model (~1GB) will be downloaded automatically if not present. Ensure you have sufficient disk space and a stable internet connection.
            """)
        
        # Show API key input for Google Gemini
        elif model_choice == "Remote (Google Gemini)":
            st.markdown("### API Key Configuration")
            
            if not st.session_state.google_api_key:
                api_key_input = st.text_input(
                    "Google API Key",
                    type="password",
                    placeholder="Enter your Google API Key",
                    help="Required for using Google Gemini model"
                )
                
                if api_key_input:
                    st.session_state.google_api_key = api_key_input
            else:
                st.success("✅ API Key configured")
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
                        if st.button("Save"):
                            if new_api_key:
                                st.session_state.google_api_key = new_api_key
                                st.session_state.show_api_key_input = False
                                st.success("API Key updated!")
                                st.rerun()
                            else:
                                st.error("Please enter a valid API key")
                    with col2:
                        if st.button("Cancel"):
                            st.session_state.show_api_key_input = False
                            st.rerun()
        
        # Only show initialize button if we have required credentials
        can_initialize = (
            model_choice == "Local (Ollama)" or 
            (model_choice == "Remote (Google Gemini)" and st.session_state.google_api_key)
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
                        st.error(f"Failed to initialize agent: {str(e)}")
        elif model_choice == "Remote (Google Gemini)":
            st.warning("⚠️ Please enter your Google API Key to continue")
    else:
        st.success("✅ Agent Ready")
        
        # Show update API key option only when using Google Gemini
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
                    if st.button("Save", key="save_running"):
                        if new_api_key:
                            st.session_state.google_api_key = new_api_key
                            st.session_state.show_api_key_input = False
                            st.info("API Key updated. Reset agent to use the new key.")
                            st.rerun()
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
            st.rerun()
    
    st.markdown("---")
    st.markdown("""
    ### About
    This agent helps you generate Docker commands based on natural language queries.
    
    The agent may request permission to gather context from your Docker environment.
    """)

# Main chat interface
st.title("DockerGPT")

if not st.session_state.model_selected:
    st.info("👈 Please select and initialize a model from the sidebar to begin.")
else:
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Handle pending approval requests
    if st.session_state.pending_approval:
        approval_data = st.session_state.pending_approval
        
        st.warning("⚠️ The agent is requesting permission to execute a command")
        
        for call in approval_data["requests"].approvals:
            with st.container():
                st.markdown(f"**Tool:** `{call.tool_name}`")
                st.code(str(call.args), language="python")
                
                col1, col2 = st.columns([1, 1])
                with col1:
                    if st.button("✅ Approve", key=f"approve_{call.tool_call_id}", type="primary", use_container_width=True):
                        approval_data["approval_results"].approvals[call.tool_call_id] = True
                        # Store the command for display later (handle both dict and string args)
                        if isinstance(call.args, dict):
                            st.session_state.last_approved_command = call.args.get('command', '')
                        else:
                            st.session_state.last_approved_command = str(call.args)
                        st.session_state.pending_approval = None
                        
                        # Continue agent execution
                        with st.spinner("Processing..."):
                            result = run_agent_async(
                                st.session_state.agent.run,
                                message_history=approval_data["messages"],
                                deferred_tool_results=approval_data["approval_results"]
                            )
                            st.session_state.result = result
                        st.rerun()
                
                with col2:
                    if st.button("❌ Deny", key=f"deny_{call.tool_call_id}", use_container_width=True):
                        approval_data["approval_results"].approvals[call.tool_call_id] = ToolDenied(
                            "User has denied the request to execute the command. You must end this run immediately."
                        )
                        st.session_state.pending_approval = None
                        
                        # End the run immediately with "Goodbye!" message
                        st.session_state.messages.append({"role": "assistant", "content": "Goodbye!"})
                        st.session_state.result = None
                        st.rerun()
    
    # Display result if available
    if st.session_state.result:
        result = st.session_state.result
        
        if isinstance(result.output, str):
            # Check if there was a gather_context call and display its output
            if st.session_state.last_approved_command:
                # Find the tool result in the message history
                for msg in result.all_messages():
                    if hasattr(msg, 'parts'):
                        for part in msg.parts:
                            if hasattr(part, 'tool_name') and part.tool_name == 'gather_context':
                                if hasattr(part, 'content') and part.content:
                                    # Display context in chat
                                    context_message = f"**🔍 Context gathered from command:** `{st.session_state.last_approved_command}`\n```bash\n{part.content}\n```"
                                    st.session_state.messages.append({"role": "system", "content": context_message})
                                    st.session_state.last_approved_command = None
                                    break
            
            st.session_state.messages.append({"role": "assistant", "content": result.output})
            
            # Display usage info
            with st.expander("📊 Usage Statistics"):
                usage = result.usage()
                st.json({
                    "requests": usage.requests,
                    "request_tokens": usage.request_tokens,
                    "response_tokens": usage.response_tokens,
                    "total_tokens": usage.total_tokens,
                    "details": str(usage.details) if usage.details else None
                })
            
            st.session_state.result = None
            st.rerun()
    
    # Chat input
    if prompt := st.chat_input("Ask for a Docker command...", disabled=st.session_state.pending_approval is not None):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get agent response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = run_agent_async(st.session_state.agent.run, user_prompt=prompt)
                
                # Check if we need approval
                if isinstance(result.output, DeferredToolRequests):
                    st.session_state.pending_approval = {
                        "requests": result.output,
                        "messages": result.all_messages(),
                        "approval_results": DeferredToolResults()
                    }
                    st.rerun()
                else:
                    st.session_state.result = result
                    st.rerun()