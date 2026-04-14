import csv
import os

# -------- CONFIG --------
NUM_SHARDS = 500
INPUT_FILE = "neighbors.csv"
OUTPUT_DIR = "shards"
MAX_NEIGHBOURS = 20
# ------------------------


def shard_id(word, num_shards):
    """Deterministic string hash (must match frontend!)"""
    word = word.lower().strip()
    h = 0
    for c in word:
        h = (h * 31 + ord(c)) % num_shards
    return h


def main():
    print("Starting sharding process...")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Dynamically build header
    header = ["lemma"] + [f"n{i+1}" for i in range(MAX_NEIGHBOURS)]

    # Create file handles for each shard
    shard_files = {}
    shard_writers = {}

    for i in range(NUM_SHARDS):
        filename = os.path.join(OUTPUT_DIR, f"shard_{i:02d}.csv")
        f = open(filename, "w", newline="", encoding="utf-8")
        writer = csv.writer(f)

        writer.writerow(header)

        shard_files[i] = f
        shard_writers[i] = writer

    # Read input and distribute rows
    with open(INPUT_FILE, encoding="utf-8") as infile:
        reader = csv.DictReader(infile)

        count = 0

        for row in reader:
            lemma = row["lemma"]
            sid = shard_id(lemma, NUM_SHARDS)

            # Collect neighbours dynamically
            neighbours = [
                row.get(f"n{i+1}", "") for i in range(MAX_NEIGHBOURS)
            ]

            shard_writers[sid].writerow([lemma] + neighbours)

            count += 1

            if count % 1000 == 0:
                print(f"Processed {count} rows...")

    # Close all files
    for f in shard_files.values():
        f.close()

    print("\nDone!")
    print(f"Sharded {count} rows into {NUM_SHARDS} files")
    print(f"Output directory: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()