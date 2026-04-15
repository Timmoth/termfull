import csv
import os

# -------- CONFIG --------
INPUT_FILE = "neighbors.csv"
OUTPUT_DIR = "shards"
MAX_NEIGHBOURS = 20
MAX_FREQ = 100
NUM_SHARDS = MAX_FREQ + 1
# ------------------------


def parse_freq(value):
    value = str(value).strip()
    if not value:
        return 0
    try:
        return int(float(value))
    except ValueError:
        return 0


def main():
    print("Starting sharding process...")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    header = ["word", "freq"] + [f"n{i+1}" for i in range(MAX_NEIGHBOURS)]

    shard_files = {}
    shard_writers = {}

    for i in range(NUM_SHARDS):
        filename = os.path.join(OUTPUT_DIR, f"shard_{i:03d}.csv")
        f = open(filename, "w", newline="", encoding="utf-8")
        writer = csv.writer(f)
        writer.writerow(header)

        shard_files[i] = f
        shard_writers[i] = writer

    count = 0

    with open(INPUT_FILE, encoding="utf-8") as infile:
        reader = csv.DictReader(infile)

        for row in reader:
            word = row["word"].strip()
            freq = parse_freq(row.get("freq", 0))
            freq = max(0, min(freq, MAX_FREQ))

            neighbours = [row.get(f"n{i+1}", "") for i in range(MAX_NEIGHBOURS)]

            shard_writers[freq].writerow([word, freq] + neighbours)

            count += 1

            if count % 1000 == 0:
                print(f"Processed {count} rows...")

    for f in shard_files.values():
        f.close()

    print("\nDone!")
    print(f"Sharded {count} rows into {NUM_SHARDS} files")
    print(f"Output directory: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()