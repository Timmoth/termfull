import csv
import numpy as np
import re
from sentence_transformers import SentenceTransformer

# -------- CONFIG --------
TOP_K = 20
BATCH_SIZE = 256
MODEL_NAME = "BAAI/bge-base-en-v1.5"

CANDIDATE_POOL = TOP_K * 15
FREQ_WEIGHT = 0.3
DIVERSITY_WEIGHT = 0.25  # MMR strength

SEARCH_BATCH_SIZE = 512

LIMIT = None  # e.g. 10_000 to cap rows
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


# -------- NORMALIZATION --------
_punct_re = re.compile(r"[^\w\s]")

def clean_word(word):
    """For embeddings (only remove punctuation)"""
    return _punct_re.sub("", word.lower()).strip()


def dedupe_key(word):
    """For deduplication (light normalization)"""
    word = clean_word(word)

    if word.endswith("ly"):
        word = word[:-2]
    elif word.endswith("ity"):
        word = word[:-3]
    elif word.endswith("er"):
        word = word[:-2]
    elif word.endswith("est"):
        word = word[:-3]

    return word


# -------- LOAD --------
def load_words_and_freqs():
    word_freqs = {}

    with open("terms.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if LIMIT and i >= LIMIT:
                break

            word = row["word"].strip()
            if not word:
                continue

            raw_freq = row.get("frequency", "").strip()
            freq = 0.0
            if raw_freq:
                try:
                    freq = float(raw_freq)
                except ValueError:
                    pass

            if word not in word_freqs or freq > word_freqs[word]:
                word_freqs[word] = freq

    words = sorted(word_freqs.keys())
    print(f"Loaded {len(words)} words")
    return words, word_freqs


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

    # amplify distribution slightly
    logged = np.log1p(raw) ** 1.3

    max_val = logged.max() if len(logged) else 1.0
    if max_val <= 0:
        return np.zeros_like(logged)

    return logged / max_val


# -------- MMR SELECTION --------
def mmr_select(i, sims_row, candidate_indices, embeddings, freq_bonus, words):
    selected = []
    selected_embs = []
    seen = set()

    base_key = dedupe_key(words[i])
    seen.add(base_key)

    candidates = list(candidate_indices)

    while candidates and len(selected) < TOP_K:
        best_idx = None
        best_score = -1e9

        for idx in candidates:
            key = dedupe_key(words[idx])
            if key in seen:
                continue

            sim_score = sims_row[idx]
            freq_score = freq_bonus[idx] * FREQ_WEIGHT

            # diversity penalty
            if selected_embs:
                sims_to_selected = embeddings[idx] @ np.stack(selected_embs).T
                diversity_penalty = sims_to_selected.max()
            else:
                diversity_penalty = 0.0

            score = sim_score + freq_score - (diversity_penalty * DIVERSITY_WEIGHT)

            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx is None:
            break

        selected.append(best_idx)
        selected_embs.append(embeddings[best_idx])
        seen.add(dedupe_key(words[best_idx]))

        candidates.remove(best_idx)

    return [words[idx] for idx in selected]


# -------- NEIGHBORS --------
def compute_top_neighbors(words, embeddings, word_freqs):
    print("Computing neighbors with MMR...")

    n = len(words)
    neighbors = []

    freq_bonus = build_frequency_bonus(words, word_freqs)

    for start in range(0, n, SEARCH_BATCH_SIZE):
        end = min(start + SEARCH_BATCH_SIZE, n)
        batch = embeddings[start:end]

        sims_batch = batch @ embeddings.T

        # remove self similarity
        row_indices = np.arange(end - start)
        sims_batch[row_indices, start + row_indices] = -1.0

        top_idx_batch = np.argpartition(
            sims_batch, -CANDIDATE_POOL, axis=1
        )[:, -CANDIDATE_POOL:]

        for local_i, candidate_indices in enumerate(top_idx_batch):
            i = start + local_i
            word = words[i]

            sims_row = sims_batch[local_i]

            selected_words = mmr_select(
                i,
                sims_row,
                candidate_indices,
                embeddings,
                freq_bonus,
                words
            )

            row = [word] + selected_words

            if len(selected_words) < TOP_K:
                row += [""] * (TOP_K - len(selected_words))

            neighbors.append(row)

        print(f"Processed {end}/{n}")

    return neighbors


# -------- SAVE --------
def save_csv(rows):
    print("Saving CSV...")

    header = ["word"] + [f"n{i+1}" for i in range(TOP_K)]

    with open("neighbors.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print("Saved to neighbors.csv")


# -------- MAIN --------
def main():
    words, word_freqs = load_words_and_freqs()
    embeddings = generate_embeddings(words)
    rows = compute_top_neighbors(words, embeddings, word_freqs)
    save_csv(rows)

    print("Done!")


if __name__ == "__main__":
    main()