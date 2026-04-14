import csv
import numpy as np
from sentence_transformers import SentenceTransformer

# -------- CONFIG --------
TOP_K = 20
BATCH_SIZE = 256
MODEL_NAME = "BAAI/bge-base-en-v1.5"

# Pull more candidates so reranking has room to improve quality
CANDIDATE_POOL = TOP_K * 15

# How strongly frequency influences final ranking
FREQ_WEIGHT = 0.08

# Batch size for neighbor search
SEARCH_BATCH_SIZE = 512
# ------------------------

print("Loading model...")
model = SentenceTransformer(MODEL_NAME)

# Enable Apple GPU if available
try:
    import torch
    if torch.backends.mps.is_available():
        print("Using Apple GPU (MPS)")
        model = model.to("mps")
except Exception:
    pass


# -------- SIMPLE NORMALISATION (FAST) --------
def normalize(word):
    word = word.lower()

    if word.endswith("ing") and len(word) > 4:
        word = word[:-3]
    elif word.endswith("ed") and len(word) > 3:
        word = word[:-2]
    elif word.endswith("s") and len(word) > 3:
        word = word[:-1]

    return word


# -------- LOAD --------
def load_unique_lemmas_and_freqs():
    lemma_freqs = {}

    with open("terms.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lemma = row["lemma"].strip().lower()
            if not lemma:
                continue

            raw_freq = row.get("frequency", "").strip()
            freq = 0.0
            if raw_freq:
                try:
                    freq = float(raw_freq)
                except ValueError:
                    freq = 0.0

            # Use max frequency seen for the lemma
            if lemma not in lemma_freqs or freq > lemma_freqs[lemma]:
                lemma_freqs[lemma] = freq

    lemmas = sorted(list(lemma_freqs.keys()))
    print(f"Loaded {len(lemmas)} unique lemmas")
    return lemmas, lemma_freqs


# -------- EMBEDDINGS --------
def generate_embeddings(lemmas):
    print("Generating embeddings...")

    embeddings = model.encode(
        lemmas,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # Make sure BLAS gets a nice contiguous float32 array
    return np.ascontiguousarray(embeddings.astype(np.float32))


# -------- FREQUENCY SCORING --------
def build_frequency_bonus(lemmas, lemma_freqs):
    raw = np.array([lemma_freqs.get(lemma, 0.0) for lemma in lemmas], dtype=np.float32)

    logged = np.log1p(raw)

    max_val = logged.max() if len(logged) else 1.0
    if max_val <= 0:
        return np.zeros_like(logged)

    return logged / max_val


# -------- NEIGHBORS --------
def compute_top_neighbors(lemmas, embeddings, lemma_freqs):
    print("Computing neighbors (batched, same logic)...")

    n = len(lemmas)
    neighbors = []

    freq_bonus = build_frequency_bonus(lemmas, lemma_freqs)
    normalized_lemmas = [normalize(lemma) for lemma in lemmas]

    for start in range(0, n, SEARCH_BATCH_SIZE):
        end = min(start + SEARCH_BATCH_SIZE, n)
        batch = embeddings[start:end]  # shape: [B, D]

        # One big batch dot-product instead of one per row
        sims_batch = batch @ embeddings.T  # shape: [B, N]

        # Exclude self for each row in batch
        row_indices = np.arange(end - start)
        sims_batch[row_indices, start + row_indices] = -1.0

        # Get semantic candidate pools for the whole batch
        top_idx_batch = np.argpartition(sims_batch, -CANDIDATE_POOL, axis=1)[:, -CANDIDATE_POOL:]

        for local_i, top_indices in enumerate(top_idx_batch):
            i = start + local_i
            lemma = lemmas[i]

            # Rerank candidates by semantic score + frequency prior
            candidate_sims = sims_batch[local_i, top_indices]
            rerank_scores = candidate_sims + (freq_bonus[top_indices] * FREQ_WEIGHT)
            top_indices = top_indices[np.argsort(rerank_scores)[::-1]]

            # -------- DEDUPE --------
            seen = set()
            row_neighbors = []

            base_key = normalized_lemmas[i]
            seen.add(base_key)

            for idx in top_indices:
                key = normalized_lemmas[idx]

                if key in seen:
                    continue

                seen.add(key)
                row_neighbors.append(lemmas[idx])

                if len(row_neighbors) >= TOP_K:
                    break

            row = [lemma] + row_neighbors

            if len(row_neighbors) < TOP_K:
                row += [""] * (TOP_K - len(row_neighbors))

            neighbors.append(row)

        print(f"Processed {end}/{n}")

    return neighbors


# -------- SAVE --------
def save_csv(rows):
    print("Saving CSV...")

    header = ["lemma"] + [f"n{i+1}" for i in range(TOP_K)]

    with open("neighbors.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print("Saved to neighbors.csv")


# -------- MAIN --------
def main():
    lemmas, lemma_freqs = load_unique_lemmas_and_freqs()
    embeddings = generate_embeddings(lemmas)
    rows = compute_top_neighbors(lemmas, embeddings, lemma_freqs)
    save_csv(rows)

    print("Done!")


if __name__ == "__main__":
    main()