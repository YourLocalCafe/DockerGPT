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

# Load environment variables
load_dotenv()

# Configure logfire
logfire.configure(token=getenv("LOGFIRE_WRITE_TOKEN"))
logfire.instrument_pydantic_ai()

# Page config
st.set_page_config(page_title="Docker Command Agent", page_icon="🐳", layout="wide")

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

def initialize_agent(model_choice):
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
                'The final command is to be given to the user directly as a string in the response, you do not have to give the user the output of the command'
            ],
            output_type=[str, DeferredToolRequests]
        )
    else:  # Remote (Google)
        model = GoogleModel(
            'gemini-2.5-flash',
            provider=GoogleProvider(api_key=getenv("GOOGLE_API_KEY"))
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
    st.title("🐳 Docker Agent")
    st.markdown("---")
    
    if not st.session_state.model_selected:
        st.subheader("Model Selection")
        model_choice = st.radio(
            "Choose your model:",
            ["Local (Ollama)", "Remote (Google Gemini)"],
            help="Local model uses Ollama with qwen3:1.7b. Remote uses Google Gemini 2.5 Flash."
        )
        
        if st.button("Initialize Agent", type="primary"):
            with st.spinner("Initializing agent..."):
                try:
                    st.session_state.agent = initialize_agent(model_choice)
                    st.session_state.model_selected = True
                    st.success(f"Agent initialized with {model_choice}!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to initialize agent: {str(e)}")
    else:
        st.success("✅ Agent Ready")
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
st.title("Docker Command Agent")

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
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("✅ Approve", key=f"approve_{call.tool_call_id}", type="primary"):
                        approval_data["approval_results"].approvals[call.tool_call_id] = True
                        st.session_state.pending_approval = None
                        
                        # Continue agent execution
                        with st.spinner("Processing..."):
                            result = st.session_state.agent.run_sync(
                                message_history=approval_data["messages"],
                                deferred_tool_results=approval_data["approval_results"]
                            )
                            st.session_state.result = result
                        st.rerun()
                
                with col2:
                    if st.button("❌ Deny", key=f"deny_{call.tool_call_id}"):
                        approval_data["approval_results"].approvals[call.tool_call_id] = ToolDenied(
                            "User has denied the request to execute the command. You must end this run immediately."
                        )
                        st.session_state.pending_approval = None
                        
                        # Continue agent execution with denial
                        with st.spinner("Processing..."):
                            result = st.session_state.agent.run_sync(
                                message_history=approval_data["messages"],
                                deferred_tool_results=approval_data["approval_results"]
                            )
                            st.session_state.result = result
                        st.rerun()
    
    # Display result if available
    if st.session_state.result:
        result = st.session_state.result
        
        if isinstance(result.output, str):
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
                result = st.session_state.agent.run_sync(user_prompt=prompt)
                
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