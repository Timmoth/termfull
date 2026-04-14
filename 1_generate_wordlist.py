from wordfreq import top_n_list, zipf_frequency
import csv

# -------- CONFIG --------
MAX_WORDS = 2000000
MIN_FREQUENCY = 0
MIN_LENGTH = 3
# ------------------------


def is_valid_word(word):
    return word.isalpha()


def main():
    print("Loading word list...")

    words = top_n_list("en", MAX_WORDS)

    print(f"Processing {len(words)} words...")

    kept = 0

    with open("words_freq.csv", "w", newline="", encoding="utf-8") as f_freq, \
         open("words.csv", "w", newline="", encoding="utf-8") as f_words:

        writer_freq = csv.writer(f_freq)
        writer_words = csv.writer(f_words)

        # headers
        writer_freq.writerow(["word", "frequency"])
        writer_words.writerow(["word"])

        for word in words:
            word = word.lower()

            if not is_valid_word(word):
                continue

            if len(word) < MIN_LENGTH:
                continue

            freq = zipf_frequency(word, "en")

            if freq < MIN_FREQUENCY:
                continue

            # full version
            writer_freq.writerow([word, round(freq, 3)])

            # lightweight version
            writer_words.writerow([word])

            kept += 1

            if kept % 1000 == 0:
                print(f"Kept {kept} words...")

    print(f"\nDone!")
    print(f"Saved {kept} words to:")
    print(" - words_freq.csv")
    print(" - words.csv")


if __name__ == "__main__":
    main()