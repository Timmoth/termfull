import csv
import spacy

print("Loading spaCy model...")
nlp = spacy.load(
    "en_core_web_sm",
    disable=["parser", "ner", "textcat"]
)

# Store lemma -> ID
lemma_to_id = {}

# Track duplicates
seen_words = set()

BATCH_SIZE = 2000
N_PROCESS = 1  # set to -1 if you want multiprocessing


def parse_frequency(row):
    """
    Accept either 'frequency' or 'quantized_score' from the input CSV.
    Returns an integer if possible, otherwise empty string.
    """
    raw = (
        row.get("freq", "").strip()
    )

    if not raw:
        return ""

    try:
        return int(float(raw))
    except ValueError:
        return ""


def main():
    global lemma_to_id

    current_id = 0
    rows_written = 0

    print("Reading words...")

    items = []

    with open("words_freq.csv", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)

        for row in reader:
            word = row["word"].strip().lower()
            if not word:
                continue

            if word in seen_words:
                continue
            seen_words.add(word)

            items.append({
                "word": word,
                "frequency": parse_frequency(row),
            })

    print(f"Processing {len(items)} unique words...")

    with open("terms.csv", "w", newline="", encoding="utf-8") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["id", "word", "lemma", "frequency"])

        words = [item["word"] for item in items]

        for i, doc in enumerate(
            nlp.pipe(words, batch_size=BATCH_SIZE, n_process=N_PROCESS)
        ):
            if not doc:
                continue

            token = doc[0]
            word = items[i]["word"]
            frequency = items[i]["frequency"]

            lemma = token.lemma_.lower().strip()

            # Skip weird outputs
            if not lemma.isalpha():
                continue

            if lemma not in lemma_to_id:
                lemma_to_id[lemma] = current_id
                current_id += 1

            lemma_id = lemma_to_id[lemma]

            writer.writerow([lemma_id, word, lemma, frequency])
            rows_written += 1

            if rows_written % 1000 == 0:
                print(f"Processed {rows_written} words...")

    print("\nDone!")
    print(f"Total lemmas: {len(lemma_to_id)}")
    print(f"Total rows: {rows_written}")


if __name__ == "__main__":
    main()