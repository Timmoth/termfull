import csv
import spacy

print("Loading spaCy model...")
# Keep tagger + lemmatizer for POS + lemma, disable anything else
nlp = spacy.load(
    "en_core_web_sm",
    disable=["parser", "ner", "textcat"]
)

# Store lemma → ID
lemma_to_id = {}

# Track duplicates
seen_words = set()

# Tune these for your machine / dataset size
BATCH_SIZE = 2000
N_PROCESS = 1  # set to -1 to use all CPU cores if that works well on your machine


def parse_frequency(row):
    value = row.get("frequency", "").strip()
    if not value:
        return ""

    try:
        return float(value)
    except ValueError:
        return ""


def main():
    global lemma_to_id

    current_id = 0
    rows_written = 0

    print("Reading words...")

    # First collect unique words in input order, keeping frequency
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
        writer.writerow(["id", "word", "lemma", "pos", "frequency"])

        words = [item["word"] for item in items]

        for i, doc in enumerate(
            nlp.pipe(words, batch_size=BATCH_SIZE, n_process=N_PROCESS)
        ):
            token = doc[0]
            word = items[i]["word"]
            frequency = items[i]["frequency"]

            lemma = token.lemma_
            pos = token.pos_.lower()

            # Skip weird outputs
            if not lemma.isalpha():
                continue

            if lemma not in lemma_to_id:
                lemma_to_id[lemma] = current_id
                current_id += 1

            lemma_id = lemma_to_id[lemma]

            writer.writerow([lemma_id, word, lemma, pos, frequency])
            rows_written += 1

            if rows_written % 1000 == 0:
                print(f"Processed {rows_written} words...")

    print("\nDone!")
    print(f"Total lemmas: {len(lemma_to_id)}")
    print(f"Total rows: {rows_written}")


if __name__ == "__main__":
    main()