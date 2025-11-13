import subprocess
import re
import chromadb
import os
import shutil


DB_PATH = "./chroma_docker_db"
COLLECTION_NAME = "docker_help"

def get_main_subcommands() -> list[str]:
    """
    Runs 'docker --help' 
    """
    print("Fetching main 'docker --help' to find subcommands...")
    try:
        result = subprocess.run(
            "docker --help", 
            shell=True, capture_output=True, text=True, check=True, encoding='utf-8'
        )
        output = result.stdout
        
        commands = []
        
        
        for section in re.findall(r"(?:Commands|Management Commands):\n([\s\S]*?)(?:\n\n|\Z)", output):
            
            for line in section.strip().split('\n'):
                
                match = re.match(r"^\s+([a-zA-Z0-9-\*]+)", line)
                if match:
                    
                    command_name = match.group(1).replace('*', '') 
                    if command_name: 
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
    
    command = f"docker {command_name} --help" if command_name else "docker --help"
    try:
        
        result = subprocess.run(
            command, 
            shell=True, 
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True, 
            encoding='utf-8'
        )
        
        
        if result.stdout:
            return result.stdout
        
        
        
        if result.stderr:
            return result.stderr
            
        
        return ""
        
    except Exception as e:
        
        print(f"Error fetching help for '{command}': {e}")
        return ""

def chunk_help_text(command_name: str | None, help_text: str) -> tuple[list, list, list]:
    """
    Creates a single chunk containing the entire help text for a command.
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
    
    collection = client.get_or_create_collection(COLLECTION_NAME)
    
    
    
    commands_to_process = [None] 
    commands_to_process.extend(get_main_subcommands())
    
    if len(commands_to_process) == 1: 
        print("No subcommands found. Halting.")
        return

    total_chunks = 0
    for i, cmd in enumerate(commands_to_process):
        command_display = cmd if cmd else "docker (main)"
        print(f"\n--- Processing ({i+1}/{len(commands_to_process)}): {command_display} ---")
        

        help_text = get_help_text(cmd)
        if not help_text.strip():
            print(f"Skipping {command_display}, no help text found.")
            continue
            
        
        docs, metas, ids = chunk_help_text(cmd, help_text)
        
        if not docs:
            print("No chunks generated.")
            continue
            
        print(f"Generated {len(docs)} document chunk (complete help text).")
        
        
        try:
            
            .
            collection.upsert(documents=docs, metadatas=metas, ids=ids)
            total_chunks += len(docs)
        except Exception as e:
            print(f"Error adding to ChromaDB: {e}")
            print("This can happen with duplicate IDs from parsing. Check parser if errors persist.")

    print(f"\n Successfully created ChromaDB collection '{COLLECTION_NAME}'")
    print(f"   at '{DB_PATH}' with {total_chunks} document chunks.")
    return collection

def query_db(collection, query_text: str, n: int = 3):
    """
    Helper function to test the database with a query.
    """
    print(f"\n Querying for: '{query_text}'")
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


if __name__ == "__main__":
    
    docker_collection = create_docker_help_db()
    
    
    if docker_collection:
        query_db(docker_collection, "how to run a container in the background")
        query_db(docker_collection, "what does the --rm flag do on docker run")
        query_db(docker_collection, "how do I list all containers including stopped ones")
        query_db(docker_collection, "how to build an image from a dockerfile")