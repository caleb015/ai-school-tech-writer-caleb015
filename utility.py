import os
import base64
import numpy as np
import faiss
import logging
import openai
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from github import Github, GithubException

load_dotenv()

base_branch = "master"

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize FAISS index
index = None
embedding_data = []
embedding_metadata = []  # List to store metadata

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)

def get_embedding(text, model="text-embedding-3-large"):
    """Get embeddings using the updated OpenAI API."""
    try:
        chunks = text_splitter.split_text(text)
        embeddings = []
        for chunk in chunks:
            response = client.embeddings.create(input=[chunk], model=model)
            embeddings.append(response.data[0].embedding)
        return embeddings
    except Exception as e:
        logging.error(f"Error getting embedding: {e}")
        return None

def initialize_index(dimension):
    global index
    index = faiss.IndexFlatL2(dimension)
    logging.info(f"Initialized FAISS index with dimension {dimension}")

def embed_and_store(commit_messages, commit_diffs, commit_ids, file_names, embeddings_file='embeddings_output.txt'):
    global index, embedding_data, embedding_metadata
    
    combined_data = commit_messages + commit_diffs
    
    with open(embeddings_file, 'w') as ef:
        for i, text in enumerate(combined_data):
            embeddings = get_embedding(text)
            if embeddings is None or len(embeddings) == 0:
                logging.warning(f"Failed to get embedding for item {i+1}/{len(combined_data)}. Skipping.")
                ef.write(f"Failed to get embedding for item {i+1}/{len(combined_data)}. Skipping.\n")
                continue
            
            # Initialize index with the dimension of the first embedding we receive
            if index is None:
                dimension = len(embeddings[0])
                initialize_index(dimension)
            
            for j, embedding in enumerate(embeddings):
                embedding_np = np.array(embedding).reshape(1, -1)

                # Check if the shape is correct
                if embedding_np.shape[1] != index.d:
                    logging.warning(f"Embedding dimension mismatch. Expected {index.d}, got {embedding_np.shape[1]}. Skipping.")
                    ef.write(f"Embedding dimension mismatch. Expected {index.d}, got {embedding_np.shape[1]}. Skipping.\n")
                    continue

                logging.info(f"Embedding shape: {embedding_np.shape}")

                # Add the embedding to the FAISS index
                index.add(embedding_np)
                embedding_data.append(text)
                
                metadata = {
                    "type": "commit_message" if i < len(commit_messages) else "commit_diff",
                    "file_name": file_names[i % len(file_names)],
                    "commit_id": commit_ids[i % len(commit_ids)],
                    "chunk_index": j
                }
                embedding_metadata.append(metadata)
                
                # Write only the text chunk and metadata to the file
                ef.write(f"Embedding {i+1}, Chunk {j+1}:\n")
                ef.write(f"Text: {text}\n")
                ef.write(f"Metadata: {metadata}\n\n")
            
            logging.info(f"Processed item {i+1}/{len(combined_data)}")
    
    logging.info(f"Stored {index.ntotal} embeddings in the vector database.")

def retrieve_relevant_data(current_changes, top_k=20):
    global index, embedding_data, embedding_metadata
    
    if index is None or index.ntotal == 0:
        logging.warning("No embeddings in the database. Returning empty result.")
        return [], []
    
    logging.info("Retrieving relevant data from the vector database.")
    
    query_embeddings = get_embedding(current_changes)
    if query_embeddings is None or len(query_embeddings) == 0:
        logging.warning("Failed to get query embedding. Returning empty result.")
        return [], []
    
    all_distances = []
    all_indices = []
    
    for query_embedding in query_embeddings:
        query_embedding_np = np.array(query_embedding).reshape(1, -1)
        if query_embedding_np.shape[1] != index.d:
            logging.warning(f"Query embedding dimension mismatch. Expected {index.d}, got {query_embedding_np.shape[1]}. Skipping.")
            continue
        distances, indices = index.search(query_embedding_np, top_k)
        all_distances.extend(distances[0])
        all_indices.extend(indices[0])
    
    # Sort by distance and get top_k unique indices
    sorted_results = sorted(zip(all_distances, all_indices), key=lambda x: x[0])
    unique_indices = []
    for _, idx in sorted_results:
        if idx not in unique_indices:
            unique_indices.append(idx)
        if len(unique_indices) == top_k:
            break
    
    relevant_data = [embedding_data[i] for i in unique_indices]
    relevant_metadata = [embedding_metadata[i] for i in unique_indices]
    
    logging.info(f"Retrieved {len(relevant_data)} relevant items from the vector database.")
    
    return relevant_data, relevant_metadata

def format_data_for_openai(metadata, readme_content, commit_messages):
    logging.info("Formatting data for OpenAI prompt.")
    
    # Decode the README content
    readme = base64.b64decode(readme_content.content).decode("utf-8")
    
    # Format metadata
    formatted_metadata = "\n".join([
        f"Type: {item.get('type', 'N/A')}, File: {item.get('file_name', 'N/A')}, Commit: {item.get('commit_id', 'N/A')}"
        for item in metadata if isinstance(item, dict)
    ])
    
    # Combine all commit messages
    messages = "\n".join(commit_messages)
    
    prompt = (
    "Please review the following information from a GitHub repository:\n\n"
    f"Relevant changes:\n{formatted_metadata}\n\n"
    f"Commit messages:\n{messages}\n\n"
    "Current README content:\n"
    f"{readme}\n\n"
    "Based on the relevant changes and commit messages, determine if the README needs to be updated. "
    "If so, edit the README to accurately reflect the changes, including any updates to usage instructions, "
    "new or modified commands, or additional dependencies. Ensure that the README maintains its existing style, "
    "clarity, and formatting. Also, remove or mark as obsolete any deprecated features or instructions. "
    "Incorporate best practices, specify required versions of tools or dependencies, and add any necessary "
    "troubleshooting tips. Finally, provide the updated README content below:\n\n"
    "Updated README:\n"
)

    
    logging.info("Data formatting complete.")
    return prompt


def call_openai(prompt: str, verbose=True) -> str:
    """Send the prompt to OpenAI, return the response."""
    client = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    try:
        logging.info("Sending prompt to OpenAI.Prompt: {prompt}")
        messages = [
            {
                "role": "system",
                "content": """
                    You are an AI trained to review commits to identify changes in the codebase, focusing on updates to application logic, 
                    feature additions or removals, configuration modifications, and dependency changes. 
                    
                    Analyze the code changes to determine their impact on the application or tool's usage, paying particular attention to 
                    new or modified commands, scripts, or functions.

                    Based on these findings, update the README file by modifying existing instructions to reflect any changes in usage, adding 
                    documentation for new commands, options, or configurations, and including sections for new dependencies with detailed setup 
                    instructions. 
                    
                    Ensure that deprecated features or instructions are removed or marked as obsolete. 
                    
                    Enhance the documentation by incorporating new best practices, specifying required versions of tools or dependencies, and 
                    adding troubleshooting tips for potential issues. 
                    
                    Maintain consistency in style and formatting throughout the README, and cross-check that all links and 
                    references remain valid and relevant. 
                """
            },
            {"role": "user", "content": prompt},
        ]

        # Call OpenAI
        response = client.invoke(input=messages)

        # If response is a list of AIMessage objects, access content directly
        if isinstance(response, list):
            content = response[0].content  # Assuming you only need the first message
        else:
            content = response.content

        if verbose:
            logging.info(f"OpenAI response: {content}")

        return content
    except Exception as e:
        logging.error(f"Error making OpenAI API call: {e}")
        return None

def update_readme_in_branch(repo, updated_readme: str, readme_sha: str, base_branch: str, create_pr: bool = False):
    """Update the README content, creating a new branch if necessary, and optionally create a PR."""
    commit_message = "Proposed README update based on recent code changes"
    pr_title = "Update README with latest changes"
    logging.info(f"Updating README based on {base_branch} branch.")
    main_branch = repo.get_branch(base_branch)
    
    # Generate a unique branch name
    new_branch_name = f"update-readme-{readme_sha[:10]}"
    
    # Check if the branch already exists
    try:
        existing_branch = repo.get_branch(new_branch_name)
        logging.warning(f"Branch '{new_branch_name}' already exists. Updating the README in the existing branch.")
        branch_name_to_use = new_branch_name
    except:
        # Create the new branch if it doesn't exist
        new_branch = repo.create_git_ref(
            ref=f"refs/heads/{new_branch_name}", sha=main_branch.commit.sha
        )
        logging.info(f"Branch '{new_branch_name}' created successfully.")
        branch_name_to_use = new_branch_name

    # Fetch the latest README file and its sha
    readme_content = repo.get_contents("README.md", ref=branch_name_to_use)
    latest_readme_sha = readme_content.sha

    # Update the README file in the selected branch
    repo.update_file(
        path="README.md",
        message=commit_message,
        content=updated_readme,
        sha=latest_readme_sha,
        branch=branch_name_to_use
    )

    logging.info(f"README.md updated in branch: {branch_name_to_use}")

    # Create a pull request if requested
    if create_pr:
        # Check if a PR with the same branch already exists
        existing_prs = repo.get_pulls(state='open', head=f"{repo.owner.login}:{branch_name_to_use}")
        for pr in existing_prs:
            if pr.title == pr_title:
                logging.info(f"A pull request with title '{pr_title}' already exists: {pr.html_url}. Skipping PR creation.")
                return
        
        pr_body = "This PR updates the README file based on recent code changes."
        try:
            pr = repo.create_pull(
                title=pr_title,
                body=pr_body,
                head=branch_name_to_use,
                base=base_branch
            )
            logging.info(f"Pull request created: {pr.html_url}")
        except GithubException as e:  # Use the correct exception name here
            logging.error(f"Failed to create pull request: {e.data['message']}")
