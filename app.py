import streamlit as st
import pandas as pd
import json
from main import load_embeddings, search_patients, query_mistral, safe_value

# ------------------------
# Load system (cached)
# ------------------------
@st.cache_resource
def load_system():
    return load_embeddings()   # returns model, index, metadata


# ------------------------
# Streamlit UI
# ------------------------
def main():
    st.set_page_config(page_title="Patient Query System", layout="wide")

    st.title("🔎 Patient Query & Analysis System")
    st.write("Search through patient dataset and generate medical summaries with **Mistral LLM**")

    # Load embeddings + index + metadata
    model, index, metadata = load_system()

    # Query input
    query = st.text_input("Enter patient query:", placeholder="e.g., pediatric glioblastoma patients")
    top_k = st.number_input("How many results to show:", min_value=1, max_value=20, value=5, step=1)

    if st.button("Search"):
        if not query.strip():
            st.warning("⚠️ Please enter a query.")
            return

        results = search_patients(query, model, index, metadata, top_k=top_k)

        if not results:
            st.error("No matches found.")
            return

        # Convert results to dataframe
        df = pd.DataFrame([r.to_dict() for r in results])

        # 🔍 Debug: show what columns exist
        #st.write("Available columns in metadata:", list(df.columns))

        # Only keep preferred columns if they exist
        preferred_cols = ["patient_id", "age", "gender", "survival_months", "vital_status"]
        available_cols = [c for c in preferred_cols if c in df.columns]

        if available_cols:
            df_display = df[available_cols]
        else:
            df_display = df  # fallback: show everything

        st.subheader("📊 Search Results")
        st.dataframe(df_display, use_container_width=True)

        # Prepare summary input for LLM
        summary_input = [r.to_dict() for r in results]
        llm_prompt = (
            f"Summarize the following patient data into a short medical analysis:\n\n"
            f"{json.dumps(summary_input, indent=2)}"
        )

        with st.spinner("🔮 Generating summary with Mistral LLM..."):
            summary = query_mistral(llm_prompt)

        st.subheader("📄 AI-Generated Summary")
        st.info(summary)

        # Option to download summary
        st.download_button(
            label="💾 Download Summary",
            data=summary,
            file_name="query_summary.txt",
            mime="text/plain"
        )


if __name__ == "__main__":
    main()
