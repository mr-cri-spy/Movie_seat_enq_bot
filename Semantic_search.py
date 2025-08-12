# Installing the necessary libraries
!pip install datasets
!pip install torch[cpu]
!pip install sentence-transformers

from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
import torch
import os

dataset = load_dataset("cnn_dailymail", '3.0.0', split="test")
dataset

df = dataset.to_pandas().sample(2000, random_state=42)
df

model = SentenceTransformer("all-MiniLM-L6-v2")
model

passage_embeddings = list(model.encode(df['highlights'].to_list(), show_progress_bar=True))
passage_embeddings[0].shape

query = "Find me some articles about technology and artificial intelligence"

query_embedding = model.encode(query)
similarities = util.cos_sim(query_embedding, passage_embeddings)

top_indices = torch.topk(similarities.flatten(), 3).indices
top_relevant_passages = [df.iloc[x.item()]['highlights'][:200] + "..." for x in top_indices]
top_relevant_passages

def find_relevant_news(query):
    # Encode the query using the same model
    query_embedding = model.encode(query)

    # Calculate the cosine similarity between the query and passage embeddings
    similarities = util.cos_sim(query_embedding, passage_embeddings)

    # Get the indices of the top 3 most similar passages
    top_indices = torch.topk(similarities.flatten(), 3).indices

    # Retrieve the summaries of the top 3 passages and truncate them to 160 characters
    top_relevant_passages = [df.iloc[x.item()]["highlights"][:160] + "..." for x in top_indices]

    return top_relevant_passages

# Example queries to explore
print(find_relevant_news("Natural disasters"))
print(find_relevant_news("Law enforcement and police"))
print(find_relevant_news("Politics, diplomacy and nationalism"))

def clear_screen():
    os.system("clear")

def interactive_search():
    print("Welcome to the Semantic News Search!\n")
    while True:
        print("Type in a topic you'd like to find articles about, and I'll do the searching! (Type 'exit' to quit)\n> ", end="")

        query = input().strip()

        if query.lower() == "exit":
            print("\nThanks for using the Semantic News Search! Have a great day!")
            break

        print("\n\tHere are 3 articles I found based on your query: \n")

        passages = find_relevant_news(query)
        for passage in passages:
            print("\n\t" + passage)

        input("\nPress Enter to continue searching...")
        clear_screen()

  # Start the interactive search
interactive_search()
