from pydantic_ai import (Agent, ModelRetry, DeferredToolRequests, DeferredToolResults, ToolDenied)
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.models.google import GoogleModel
from subprocess import run
from dotenv import load_dotenv
from os import getenv
import logfire
load_dotenv()

logfire.configure(token=getenv("LOGFIRE_WRITE_TOKEN"))
logfire.instrument_pydantic_ai()

model_choice = -1

while(model_choice != 0 and model_choice != 1):
    model_choice = int(input("Enter 0 for using a local model and 1 for using a remote model: "))

    if model_choice == 0:
        ollama_model = OpenAIChatModel(
            model_name='qwen3:1.7b',
            provider=OllamaProvider(base_url='http://localhost:11434/v1'),
        )
        agent = Agent(ollama_model, system_prompt=['The final output given to the user must only be docker command(s)', 'The final command is to be given to the user directly as a string in the response, you do not have to give the user the output of the command', 'Use the gather_context tool available to fill in required container IDs and other command specific information if required/possible.'], output_type=[str, DeferredToolRequests])
    elif model_choice == 1:
        model = GoogleModel('gemini-2.5-flash', provider=GoogleProvider(api_key=getenv("GOOGLE_API_KEY")))
        agent = Agent(model,system_prompt=['The final output given to the user must only be docker command(s)', 'The final command is to be given to the user directly as a string in the response, you do not have to give the user the output of the command', 'Use the gather_context tool available to fill in required container IDs and other command specific information if required/possible.'], output_type=[str, DeferredToolRequests])
    else:
        print("Invalid model choice.")

@agent.tool_plain(requires_approval=True)
def gather_context(command: str) -> str:
    """
    This tool MUST ONLY be used to gather context from the user's Docker environment, it isn't required to be called everytime.
    ONLY call this tool for contextual questions.
    It must never execute final commands.

    
    Args:
        command: The command that should be run in order to gather the required context from the user's machine, not supposed to be the final command.
    """
    result = run(command, shell=True, capture_output=True)

    if result.stderr:
        raise ModelRetry(f"There was an error in executing the command: {result.stderr}")
    
    return result.stdout

print("Type 'exit' or 'quit' to end the conversation.")

user_prompt = ""

while user_prompt != "exit" and user_prompt != "quit":
    user_prompt = input("Prompt: ").strip().lower()
    if user_prompt == "exit" or user_prompt == "quit":
        break
    result = agent.run_sync(user_prompt=user_prompt)
    while isinstance(result.output, DeferredToolRequests):
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