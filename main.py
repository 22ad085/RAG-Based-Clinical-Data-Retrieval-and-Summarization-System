import faiss
import pickle
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import subprocess
import json

# ------------------------
# Load Embeddings + Index
# ------------------------
def load_embeddings():
    print("🔹 Loading embeddings model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("🔹 Loading FAISS index and metadata...")
    index = faiss.read_index("patient_index.faiss")

    with open("patient_metadata.pkl", "rb") as f:
        metadata = pickle.load(f)

    print(f"✅ FAISS index loaded with {index.ntotal} vectors")
    print(f"✅ Metadata loaded with {len(metadata)} patients")

    return model, index, metadata


# ------------------------
# Safe value printer
# ------------------------
def safe_value(val):
    return "N/A" if pd.isna(val) else val


# ------------------------
# FAISS Search
# ------------------------
def search_patients(query, model, index, metadata, top_k=5):
    query_emb = model.encode([query]).astype("float32")
    D, I = index.search(query_emb, top_k)

    results = [metadata.iloc[int(i)] for i in I[0] if i < len(metadata)]
    return results


# ------------------------
# Run Ollama (Mistral)
# ------------------------
def query_mistral(prompt):
    try:
        result = subprocess.run(
            ["ollama", "run", "mistral:latest"],
            input=prompt.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        return result.stdout.decode("utf-8").strip()
    except Exception as e:
        return f"⚠️ Error running Mistral: {e}"


# ------------------------
# Main Loop
# ------------------------
def main():
    model, index, metadata = load_embeddings()

    while True:
        query = input("\nEnter your patient query (or type 'exit' to quit): ")
        if query.lower() == "exit":
            break

        try:
            top_k = int(input("How many results do you want to see? (default 5): ") or 5)
        except ValueError:
            top_k = 5

        results = search_patients(query, model, index, metadata, top_k=top_k)

        if not results:
            print("⚠️ No matches found.")
            continue

        print("\n🔎 Found matches. Top results:")
        summary_input = []
        for i, r in enumerate(results, 1):
            print(
                f"{i}. Patient {safe_value(r.get('patient_id'))} | "
                f"Age: {safe_value(r.get('age'))} | "
                f"Gender: {safe_value(r.get('gender'))} | "
                f"Survival: {safe_value(r.get('survival_months'))} months | "
                f"Vital: {safe_value(r.get('vital_status'))}"
            )
            summary_input.append(r.to_dict())

        # Send to Mistral
        llm_prompt = (
            f"Summarize the following patient data into a short medical analysis:\n\n"
            f"{json.dumps(summary_input, indent=2)}"
        )

        summary = query_mistral(llm_prompt)
        print("\n📄 Summary by Mistral LLM:\n")
        print(summary)

        # Save summary to file
        with open("query_summary.txt", "w", encoding="utf-8") as f:
            f.write(summary)
        print("\n💾 Summary saved to query_summary.txt")


if __name__ == "__main__":
    main()
