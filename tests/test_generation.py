"""
tests/test_generation.py

End-to-end test of the full RAG pipeline without the dashboard:
    retrieval → prompt building → LLM generation → structured response
"""

from retrieval.retriever import retrieve
from generation.prompt_builder import build_prompt
from generation.llm_caller import generate_response


def run_rag_query(query: str, username: str) -> dict:
    """Run a full RAG query and return the complete result."""
    print(f"\n{'='*60}")
    print(f"Query    : {query}")
    print(f"User     : {username}")
    print(f"{'='*60}")

    # Step 1: Retrieve relevant chunks
    chunks = retrieve(query, username, top_k=5)
    print(f"\n[1] Retrieved {len(chunks)} chunks")
    for i, c in enumerate(chunks, 1):
        print(f"    Chunk {i}: Page {c['metadata']['page']} | score={c['score']:.3f}")

    if not chunks:
        print("    No chunks retrieved - access denied or no relevant content.")
        return {}

    # Step 2: Build the prompt
    prompt_data = build_prompt(query, chunks, username)
    print(f"\n[2] Prompt built ({prompt_data['chunks_count']} chunks included)")

    # Step 3: Generate response
    print("\n[3] Calling Groq API...")
    result = generate_response(prompt_data["messages"])

    print(f"\n[4] Response received:")
    print(f"    Latency      : {result['latency']}s")
    print(f"    Input tokens : {result['input_tokens']}")
    print(f"    Output tokens: {result['output_tokens']}")
    print(f"    Citations    : {result['citations']}")
    print(f"\n--- ANSWER ---")
    print(result["response"])
    print(f"--------------")

    return result


if __name__ == "__main__":
    # Test 1: Normal finance query alice can answer
    run_rag_query("What was the total revenue and how did it change?", "alice")

    # Test 2: Query where answer is not in the document
    run_rag_query("What is the recipe for chocolate cake?", "alice")

    # Test 3: Bob trying to access finance content
    run_rag_query("What was the total revenue?", "bob")