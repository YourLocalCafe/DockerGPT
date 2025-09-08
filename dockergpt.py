from pydantic_ai import (Agent, ModelRetry, DeferredToolRequests, DeferredToolResults, ToolDenied)
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.models.openai import OpenAIChatModel
from subprocess import run
from dotenv import load_dotenv
from os import getenv
import logfire
load_dotenv()

logfire.configure(token=getenv("LOGFIRE_WRITE_TOKEN"))
logfire.instrument_pydantic_ai()

ollama_model = OpenAIChatModel(
    model_name='qwen3:1.7b',
    provider=OllamaProvider(base_url='http://localhost:11434/v1'),
)

agent = Agent(ollama_model, system_prompt=['The final output given to the user must only be docker command(s)', 'The final command is to be given to the user directly as a string in the response, you do not have to give the user the output of the command'], output_type=[str, DeferredToolRequests])

@agent.tool_plain(requires_approval=True)
def gather_context(command: str) -> str:
    """
    This tool MUST ONLY be used to gather context from the user's Docker environment, it isn't required to be called everytime.
    ONLY call this tool for contextual questions.
    It must never execute final commands. The parameter `command` is only for context-gathering purposes.

    
    Args:
        command: The command that should be run in order to gather the required context from the user's machine, not supposed to be the final command.
    """
    result = run(command, shell=True, capture_output=True)

    if result.stderr:
        raise ModelRetry(f"There was an error in executing the command: {result.stderr}")
    
    return result.stdout


result = agent.run_sync('What images do I currently have on my machine?')
if isinstance(result.output, DeferredToolRequests):
    approval_results = DeferredToolResults()
    messages = result.all_messages()
    requests = result.output
    for call in requests.approvals:
        can_execute = False
        user_consent = ""
        while user_consent == "":
            print(f"The model wants to call {call.tool_name} with the following args: {call.args}")
            user_consent = input("[N/y]: ").strip().lower()
            if user_consent == "y":
                can_execute = True
            elif user_consent == "n":
                can_execute = ToolDenied("User has denied the request to execute the command. You must end this run immediately.")
            else:
                user_consent = ""
        approval_results.approvals[call.tool_call_id] = can_execute
    result = agent.run_sync(message_history=messages, deferred_tool_results=approval_results)
print(result.output)
print(result.usage())