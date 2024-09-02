import os
import click
from github import Github
from utility import *
from dotenv import load_dotenv

@click.command()
@click.argument('repo_path')
@click.option('--branch', '-b', required=True, help='The branch name to analyze for commits.')
@click.option('--latest-commits', '-lc', default=0, type=int, help='The number of latest commits to consider. If 0, all commits will be considered.')
@click.option('--output-file', '-o', default='commit_contents.txt', type=str, help='File to output commit contents.')
@click.option('--pr', is_flag=True, help='Create a pull request after updating the branch.')
def main(repo_path, branch, latest_commits, output_file, pr):
    """
    Updates the README file based on the latest X number of commits in the specified branch.
    If -lc is not provided or set to 0, all commits will be considered.

    REPO_PATH: The path to the GitHub repository.
    """
    load_dotenv()
    
    # Initialize GitHub API with token
    github_token = os.getenv('GITHUB_TOKEN')
    if not github_token:
        raise EnvironmentError("GITHUB_TOKEN is not set in the environment variables.")
    g = Github(github_token)

    # Get the repo object
    repo = g.get_repo(repo_path)

    # Fetch README content (assuming README.md)
    readme_content = repo.get_contents("README.md")
    
    # Get the commits from the specified branch
    commits = repo.get_commits(sha=branch)
    
    # If latest_commits is greater than 0, slice the commits
    if latest_commits > 0:
        commits = commits[:latest_commits]

    # Extract commit messages, diffs, commit_ids, and file_names
    commit_messages = [commit.commit.message for commit in commits]
    commit_diffs = []
    commit_ids = [commit.sha for commit in commits]  # Collect commit IDs
    file_names = []

    for commit in commits:
        for file in commit.files:
            commit_diffs.append({
                "filename": file.filename,
                "patch": file.patch if file.patch else ""
            })
            file_names.append(file.filename)  # Collect file names

    # Write commit contents to a text file
    write_commit_contents_to_file(commit_messages, commit_diffs, output_file)

    # Generate and store embeddings for commit messages, diffs, commit_ids, and file_names
    embed_and_store(commit_messages, [diff['patch'] for diff in commit_diffs], commit_ids, file_names)

    # Retrieve relevant data based on current changes
    relevant_metadata = retrieve_relevant_data("\\n".join(commit_messages + [diff['patch'] for diff in commit_diffs]))

    # Format data for OpenAI prompt
    prompt = format_data_for_openai(relevant_metadata, readme_content, commit_messages)

    # Call OpenAI to generate the updated README content
    updated_readme = call_openai(prompt)

    # Update README in a branch (create new if necessary) and potentially create a PR
    update_readme_in_branch(repo, updated_readme, readme_content.sha, branch, create_pr=pr)

def write_commit_contents_to_file(commit_messages, commit_diffs, output_file):
    """Writes the commit messages and diffs to a text file."""
    with open(output_file, 'w') as f:
        f.write("Commit Messages:\n")
        for message in commit_messages:
            f.write(f"{message}\n")
        f.write("\nCommit Diffs:\n")
        for diff in commit_diffs:
            f.write(f"Filename: {diff['filename']}\n")
            f.write(f"Patch: {diff['patch']}\n\n")

if __name__ == '__main__':
    main()
