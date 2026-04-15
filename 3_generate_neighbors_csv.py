import csv
import numpy as np
import re
from sentence_transformers import SentenceTransformer

# -------- CONFIG --------
TOP_K = 20
BATCH_SIZE = 256
MODEL_NAME = "BAAI/bge-base-en-v1.5"

CANDIDATE_POOL = TOP_K * 10
SEARCH_BATCH_SIZE = 1024

FREQ_WEIGHT = 0.1
DIVERSITY_WEIGHT = 0.1
SIM_THRESHOLD = 0.6
LIMIT = None
USE_POS_FILTER = False
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


# -------- CLEANING --------
_punct_re = re.compile(r"[^\w\s]")

def clean_word(word):
    return _punct_re.sub("", word.lower()).strip()


# -------- LOAD --------
def load_words():
    word_freqs = {}
    word_pos = {}
    word_lemma = {}

    with open("terms.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for i, row in enumerate(reader):
            if LIMIT and i >= LIMIT:
                break

            word = row["word"].strip()
            if not word:
                continue

            lemma = row["lemma"].strip().lower()
            pos = row.get("pos", "").strip().lower()

            raw_freq = row.get("frequency", "").strip()
            freq = 0
            if raw_freq:
                try:
                    freq = int(float(raw_freq))
                except ValueError:
                    freq = 0

            # keep the highest freq seen for each word
            if word not in word_freqs or freq > word_freqs[word]:
                word_freqs[word] = freq
                word_pos[word] = pos
                word_lemma[word] = lemma

    # sort by frequency descending, then alphabetically for tie stability
    words = sorted(word_freqs.keys(), key=lambda w: (-word_freqs[w], w))
    print(f"Loaded {len(words)} words")

    return words, word_freqs, word_pos, word_lemma


# -------- EMBEDDINGS --------
def generate_embeddings(words):
    print("Generating embeddings...")

    clean_words = [clean_word(w) for w in words]

    embeddings = model.encode(
        clean_words,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    return np.ascontiguousarray(embeddings.astype(np.float32))


# -------- FREQUENCY --------
def build_frequency_bonus(words, word_freqs):
    raw = np.array([word_freqs.get(w, 0.0) for w in words], dtype=np.float32)

    logged = np.log1p(raw) ** 1.3

    max_val = logged.max() if len(logged) else 1.0
    return logged / max_val if max_val > 0 else np.zeros_like(logged)


# -------- FAST MMR (LEMMA DEDUPE) --------
def mmr_select_fast(
    i,
    sims_row,
    candidate_indices,
    embeddings,
    freq_bonus,
    words,
    word_pos,
    word_lemma,
):
    candidates = np.array(candidate_indices)

    # similarity filter
    sims = sims_row[candidates]
    mask = sims > SIM_THRESHOLD
    candidates = candidates[mask]

    if len(candidates) == 0:
        return []

    candidate_embs = embeddings[candidates]
    candidate_sims = sims_row[candidates]
    candidate_freq = freq_bonus[candidates]

    # base scoring
    base_scores = (candidate_sims ** 1.5) + (candidate_freq * FREQ_WEIGHT)

    selected = []
    selected_mask = np.zeros(len(candidates), dtype=bool)
    max_sim_to_selected = np.zeros(len(candidates), dtype=np.float32)

    # lemma + POS tracking
    used_lemmas = set()
    used_lemmas.add(word_lemma[words[i]])

    base_pos = word_pos.get(words[i], "")

    for _ in range(TOP_K):
        scores = base_scores - (max_sim_to_selected * DIVERSITY_WEIGHT)
        scores[selected_mask] = -1e9

        best_local_idx = np.argmax(scores)
        best_global_idx = candidates[best_local_idx]

        # block same lemma
        lemma = word_lemma[words[best_global_idx]]
        if lemma in used_lemmas:
            selected_mask[best_local_idx] = True
            continue

        # optional POS filter
        if USE_POS_FILTER:
            if word_pos.get(words[best_global_idx], "") != base_pos:
                selected_mask[best_local_idx] = True
                continue

        used_lemmas.add(lemma)
        selected.append(best_global_idx)
        selected_mask[best_local_idx] = True

        # update diversity
        new_sims = candidate_embs @ embeddings[best_global_idx]
        max_sim_to_selected = np.maximum(max_sim_to_selected, new_sims)

        if len(selected) >= TOP_K:
            break

    return [words[idx] for idx in selected]


# -------- NEIGHBORS --------
def compute_neighbors(words, embeddings, word_freqs, word_pos, word_lemma):
    print("Computing neighbors...")

    n = len(words)
    neighbors = []

    freq_bonus = build_frequency_bonus(words, word_freqs)

    for start in range(0, n, SEARCH_BATCH_SIZE):
        end = min(start + SEARCH_BATCH_SIZE, n)
        batch = embeddings[start:end]

        sims_batch = batch @ embeddings.T

        # remove self similarity
        row_idx = np.arange(end - start)
        sims_batch[row_idx, start + row_idx] = -1.0

        top_idx_batch = np.argpartition(
            sims_batch, -CANDIDATE_POOL, axis=1
        )[:, -CANDIDATE_POOL:]

        for local_i, candidate_indices in enumerate(top_idx_batch):
            i = start + local_i

            selected_words = mmr_select_fast(
                i,
                sims_batch[local_i],
                candidate_indices,
                embeddings,
                freq_bonus,
                words,
                word_pos,
                word_lemma,
            )

            row = [words[i], word_freqs.get(words[i], 0)] + selected_words

            if len(selected_words) < TOP_K:
                row += [""] * (TOP_K - len(selected_words))

            neighbors.append(row)

        print(f"Processed {end}/{n}")

    return neighbors


# -------- SAVE --------
def save_csv(rows):
    print("Saving CSV...")

    header = ["word", "freq"] + [f"n{i+1}" for i in range(TOP_K)]

    with open("neighbors.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print("Saved to neighbors.csv")


# -------- MAIN --------
def main():
    words, word_freqs, word_pos, word_lemma = load_words()
    embeddings = generate_embeddings(words)

    rows = compute_neighbors(words, embeddings, word_freqs, word_pos, word_lemma)

    save_csv(rows)
    print("Done!")


if __name__ == "__main__":
    main()