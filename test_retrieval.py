from retrieval.retriever import retrieve

print("\n--- TEST 1: Alice querying revenue (should return finance chunks) ---")
results = retrieve("what is the total revenue", "alice", top_k=3)
for r in results:
    print(f"Score: {r['score']:.3f} | Page: {r['metadata']['page']} | Category: {r['metadata']['category']}")
    print(f"Text: {r['text'][:200]}")
    print()

print("\n--- TEST 2: Bob querying revenue (should return 0 results) ---")
results = retrieve("what is the total revenue", "bob", top_k=3)
print(f"Results returned for bob: {len(results)}")
if len(results) == 0:
    print("Access control working correctly — bob blocked from finance documents.")

print("\n--- TEST 3: Admin querying something else ---")
results = retrieve("who are the board of directors", "admin", top_k=3)
for r in results:
    print(f"Score: {r['score']:.3f} | Page: {r['metadata']['page']}")
    print(f"Text: {r['text'][:200]}")
    print()