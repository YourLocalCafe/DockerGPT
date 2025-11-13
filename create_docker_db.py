import subprocess
import re
import chromadb
import os
import shutil

# --- Configuration ---
DB_PATH = "./chroma_docker_db"
COLLECTION_NAME = "docker_help"

def get_main_subcommands() -> list[str]:
    """
    Runs 'docker --help' and parses its output to find all
    available subcommands. This version is more robust.
    """
    print("Fetching main 'docker --help' to find subcommands...")
    try:
        result = subprocess.run(
            "docker --help", 
            shell=True, capture_output=True, text=True, check=True, encoding='utf-8'
        )
        output = result.stdout
        
        commands = []
        # Find all lines under "Commands:" and "Management Commands:" sections
        # This regex finds the section, then we parse it line-by-line
        for section in re.findall(r"(?:Commands|Management Commands):\n([\s\S]*?)(?:\n\n|\Z)", output):
            # For each line in that section...
            for line in section.strip().split('\n'):
                # ...find the first "word" on the line.
                # This regex matches: optional spaces, then the command (which can have '-', '*'),
                # then it stops.
                match = re.match(r"^\s+([a-zA-Z0-9-\*]+)", line)
                if match:
                    # Get command, remove the asterisk (like 'buildx*')
                    command_name = match.group(1).replace('*', '') 
                    if command_name: # Ensure it's not empty
                        commands.append(command_name)
        
        unique_commands = sorted(list(set(commands)))
        
        if not unique_commands:
            print("Warning: Could not parse any subcommands from 'docker --help'.")
            print("Please check your Docker installation.")
            print("Received output (first 500 chars):\n", output[:500] + "...")
        else:
            print(f"Found {len(unique_commands)} subcommands.")
            
        return unique_commands
        
    except subprocess.CalledProcessError:
        print("\n--- ERROR ---")
        print("Error: 'docker' command not found or failed to run.")
        print("Please ensure Docker is installed and in your system's PATH.")
        print("---------------")
        return []
    except Exception as e:
        print(f"An unexpected error occurred while finding commands: {e}")
        return []

def get_help_text(command_name: str | None) -> str:
    """
    Fetches the help text for a specific docker command.
    'None' fetches the main 'docker --help'.
    
    Handles commands that print help to stderr.
    """
    command = f"docker {command_name} --help" if command_name else "docker --help"
    try:
        # We explicitly capture stdout and stderr without using 'capture_output'
        # to avoid the argument conflict.
        # We also do NOT use 'check=True', as many help commands print
        # to stderr and return a non-zero exit code.
        result = subprocess.run(
            command, 
            shell=True, 
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True, 
            encoding='utf-8'
        )
        
        # If stdout has content, it's the primary source.
        if result.stdout:
            return result.stdout
        
        # If stdout is empty, but stderr has content, it's likely
        # a management command (like 'docker container') printing its help.
        if result.stderr:
            return result.stderr
            
        # If both are empty, something's wrong, return empty string
        return ""
        
    except Exception as e:
        # This is for other unexpected errors
        print(f"Error fetching help for '{command}': {e}")
        return ""

def chunk_help_text(command_name: str | None, help_text: str) -> tuple[list, list, list]:
    """
    Creates a single chunk containing the entire help text for a command.
    This prevents flags from being separated and mixed up between commands.
    Returns lists of documents, metadatas, and ids.
    """
    base_command = f"docker {command_name}" if command_name else "docker"
    documents, metadatas, ids = [], [], []
    
    # Store the entire help text as a single chunk
    if help_text.strip():
        documents.append(help_text.strip())
        metadatas.append({
            "command": base_command,
            "type": "complete_help"
        })
        ids.append(f"{base_command.replace(' ', '_')}_complete")
    
    return documents, metadatas, ids

def create_docker_help_db():
    """
    Main orchestration function.
    """
    if os.path.exists(DB_PATH):
        print(f"Database path '{DB_PATH}' already exists.")
        user_input = input("  Delete and re-create? (y/N): ").strip().lower()
        if user_input == 'y':
            print(f"Deleting existing database at {DB_PATH}...")
            shutil.rmtree(DB_PATH)
        else:
            print("Skipping creation. To re-build, delete the directory first.")
            client = chromadb.PersistentClient(path=DB_PATH)
            return client.get_collection(COLLECTION_NAME)

    print(f"Creating new database at: {DB_PATH}")
    client = chromadb.PersistentClient(path=DB_PATH)
    # Use 'get_or_create' for safety
    collection = client.get_or_create_collection(COLLECTION_NAME)
    
    # Get list of commands to process
    # Start with the main 'docker --help' (represented by None)
    commands_to_process = [None] 
    commands_to_process.extend(get_main_subcommands())
    
    if len(commands_to_process) == 1: # Only [None]
        print("No subcommands found. Halting.")
        return

    total_chunks = 0
    for i, cmd in enumerate(commands_to_process):
        command_display = cmd if cmd else "docker (main)"
        print(f"\n--- Processing ({i+1}/{len(commands_to_process)}): {command_display} ---")
        
        # 1. Get Help Text
        help_text = get_help_text(cmd)
        if not help_text.strip():
            print(f"Skipping {command_display}, no help text found.")
            continue
            
        # 2. Create single chunk with complete help text
        docs, metas, ids = chunk_help_text(cmd, help_text)
        
        if not docs:
            print("No chunks generated.")
            continue
            
        print(f"Generated {len(docs)} document chunk (complete help text).")
        
        # 3. Add to Chroma
        try:
            # Use 'upsert' to avoid issues with potential duplicate IDs from parsing
            # if the script is re-run without deleting.
            collection.upsert(documents=docs, metadatas=metas, ids=ids)
            total_chunks += len(docs)
        except Exception as e:
            print(f"Error adding to ChromaDB: {e}")
            print("This can happen with duplicate IDs from parsing. Check parser if errors persist.")

    print(f"\n🎉 Successfully created ChromaDB collection '{COLLECTION_NAME}'")
    print(f"   at '{DB_PATH}' with {total_chunks} document chunks.")
    return collection

def query_db(collection, query_text: str, n: int = 3):
    """
    Helper function to test the database with a query.
    """
    print(f"\n🔍 Querying for: '{query_text}'")
    results = collection.query(
        query_texts=[query_text],
        n_results=n
    )
    
    print("Results:")
    if not results['documents'] or not results['documents'][0]:
        print("  No results found.")
        return

    for i, (doc, meta, dist) in enumerate(zip(results['documents'][0], results['metadatas'][0], results['distances'][0])):
        print(f"  Result {i+1} (Distance: {dist:.4f}):")
        print(f"    Command: {meta.get('command')}")
        print(f"    Type:    {meta.get('type')}")
        print(f"    Text:    \"{doc[:150].strip()}...\"")

# --- Main Execution ---
if __name__ == "__main__":
    # 1. Create the database
    docker_collection = create_docker_help_db()
    
    # 2. Run some example queries
    if docker_collection:
        query_db(docker_collection, "how to run a container in the background")
        query_db(docker_collection, "what does the --rm flag do on docker run")
        query_db(docker_collection, "how do I list all containers including stopped ones")
        query_db(docker_collection, "how to build an image from a dockerfile")