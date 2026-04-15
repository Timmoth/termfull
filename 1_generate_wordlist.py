from wordfreq import top_n_list, zipf_frequency
import csv
from math import floor

# -------- CONFIG --------
MAX_WORDS = 2_000_000
MIN_FREQUENCY = 2
MIN_LENGTH = 3
MAX_QUANTIZED_SCORE = 100  # output scale: 0..100
# ------------------------


def is_valid_word(word: str) -> bool:
    return word.isalpha()


def quantize_by_rank(sorted_items, max_score):
    total = len(sorted_items)

    if total == 0:
        return []

    if total == 1:
        word, freq = sorted_items[0]
        return [(word, freq, max_score)]

    result = []
    for i, (word, freq) in enumerate(sorted_items):
        # descending list: first item should get max_score
        score = max_score - floor(i * max_score / (total - 1))
        result.append((word, freq, score))

    return result


def main():
    print("Loading word list...")
    words = top_n_list("en", MAX_WORDS)

    print(f"Processing {len(words)} words...")

    filtered = []
    seen = set()

    for idx, word in enumerate(words, 1):
        word = word.lower()

        if word in seen:
            continue
        seen.add(word)

        if not is_valid_word(word):
            continue

        if len(word) < MIN_LENGTH:
            continue

        freq = zipf_frequency(word, "en")

        if freq < MIN_FREQUENCY:
            continue

        filtered.append((word, freq))

        if len(filtered) % 1000 == 0:
            print(f"Kept {len(filtered)} words...")

    print("Sorting by frequency...")
    filtered.sort(key=lambda x: x[1], reverse=True)

    print(f"Quantizing to integer scale 0..{MAX_QUANTIZED_SCORE}...")
    quantized = quantize_by_rank(filtered, MAX_QUANTIZED_SCORE)

    CHUNK_SIZE = 10_000
    NUM_CHUNKS = (len(quantized) + CHUNK_SIZE - 1) // CHUNK_SIZE

    print(f"Splitting into {NUM_CHUNKS} chunks of {CHUNK_SIZE} words...")

    with open("words_freq.csv", "w", newline="", encoding="utf-8") as f_freq, \
         open("words.csv", "w", newline="", encoding="utf-8") as f_words:

        writer_freq = csv.writer(f_freq)
        writer_words = csv.writer(f_words)

        writer_freq.writerow(["word", "freq"])
        writer_words.writerow(["word"])

        for word, freq, score in quantized:
            writer_freq.writerow([word, score])
            writer_words.writerow([word])

    # Split into chunks - save quantized freq (the score, 0-100)
    for chunk_idx in range(NUM_CHUNKS):
        start = chunk_idx * CHUNK_SIZE
        end = min(start + CHUNK_SIZE, len(quantized))
        chunk_data = quantized[start:end]
        
        chunk_filename = f"words_freq_chunk_{chunk_idx:03d}.csv"
        with open(chunk_filename, "w", newline="", encoding="utf-8") as f_chunk:
            writer_chunk = csv.writer(f_chunk)
            writer_chunk.writerow(["word", "freq"])
            for word, freq, score in chunk_data:
                writer_chunk.writerow([word, score])  # Use quantized score, not raw freq
        
        print(f" - {chunk_filename} ({len(chunk_data)} words)")

    print("\nDone!")
    print(f"Saved {len(quantized)} words to:")
    print(" - words_freq.csv")
    print(" - words_freq_chunk_XXX.csv (chunks)")
    print(" - words.csv")


if __name__ == "__main__":
    main()