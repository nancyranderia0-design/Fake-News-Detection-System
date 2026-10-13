
import os
import re
import zipfile
import numpy as np
import pandas as pd
from scipy import stats

# ==========================================
# FAKE NEWS DETECTION SYSTEM
# GTU PDS - Semester 5
# Without Graphs
# ==========================================

FAKE_FILE = "Fake.csv"
TRUE_ZIP = "True.csv.zip"
OUTPUT_FILE = "news_results.csv"

SUSPICIOUS_WORDS = [
    "shocking", "miracle", "secret",
    "guaranteed", "breaking", "aliens", "forever"
]


# 1. LOAD DATASETS
def load_data():
    try:
        if not os.path.exists(FAKE_FILE):
            raise FileNotFoundError("Fake.csv not found.")

        if not os.path.exists(TRUE_ZIP):
            raise FileNotFoundError("True.csv.zip not found.")

        fake_df = pd.read_csv(FAKE_FILE)
        fake_df["actual_label"] = "FAKE"

        with zipfile.ZipFile(TRUE_ZIP, "r") as archive:
            csv_files = [
                name for name in archive.namelist()
                if name.lower().endswith(".csv")
            ]

            if not csv_files:
                raise ValueError("No CSV found inside ZIP.")

            with archive.open(csv_files[0]) as file:
                true_df = pd.read_csv(file)

        true_df["actual_label"] = "REAL"

        df = pd.concat(
            [fake_df, true_df],
            ignore_index=True
        )

        if "title" not in df.columns:
            raise ValueError("Dataset must contain a title column.")

        df = df.dropna(subset=["title"]).copy()
        df["title"] = df["title"].astype(str)
        df.reset_index(drop=True, inplace=True)

        print("Datasets loaded successfully!")
        print("Fake records:", len(fake_df))
        print("Real records:", len(true_df))
        print("Total records:", len(df))

        return df

    except Exception as e:
        print("Error loading data:", e)
        return None


# 2. TEXT PREPROCESSING
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


# 3. CALCULATE SUSPICION SCORE
def calculate_score(title):
    title = clean_text(title)
    score = 0

    for word in SUSPICIOUS_WORDS:
        if re.search(r"\b" + re.escape(word) + r"\b", title):
            score += 1

    return score


# 4. CLASSIFY NEWS
def detect_news(title):
    if calculate_score(title) >= 2:
        return "FAKE"
    return "REAL"


# 5. PROCESS DATA
def process_data(df):
    df = df.copy()
    df["clean_title"] = df["title"].apply(clean_text)
    df["score"] = df["title"].apply(calculate_score)
    df["prediction"] = df["title"].apply(detect_news)
    return df


# 6. READ OPERATION
def display_news(df):
    if df.empty:
        print("No records available.")
        return

    print(df[
        ["title", "actual_label", "score", "prediction"]
    ].head(20).to_string(index=True))

    print("\nShowing up to 20 records.")


# 7. CREATE OPERATION
def add_news(df):
    title = input("Enter a new headline: ").strip()

    if not title:
        print("Headline cannot be empty.")
        return df

    new_row = pd.DataFrame([{
        "title": title,
        "actual_label": "UNVERIFIED"
    }])

    df = pd.concat([df, new_row], ignore_index=True)
    print("Headline added successfully!")

    return process_data(df)


# 8. UPDATE OPERATION
def update_news(df):
    try:
        index = int(input("Enter row index to update: "))

        if index not in df.index:
            print("Invalid row index.")
            return df

        title = input("Enter updated headline: ").strip()

        if not title:
            print("Headline cannot be empty.")
            return df

        df.loc[index, "title"] = title
        print("Headline updated successfully!")

        return process_data(df)

    except ValueError:
        print("Enter a valid integer index.")
        return df


# 9. DELETE OPERATION
def delete_news(df):
    try:
        index = int(input("Enter row index to delete: "))

        if index not in df.index:
            print("Invalid row index.")
            return df

        df = df.drop(index=index).reset_index(drop=True)
        print("Headline deleted successfully!")

        return process_data(df)

    except ValueError:
        print("Enter a valid integer index.")
        return df


# 10. NUMPY ANALYSIS
def numpy_analysis(df):
    scores = df["score"].to_numpy(dtype=float)

    if len(scores) == 0:
        print("No data available.")
        return

    print("\n========== NUMPY ANALYSIS ==========")
    print("Mean score:", np.mean(scores))
    print("Maximum score:", np.max(scores))
    print("Minimum score:", np.min(scores))
    print("Standard deviation:", np.std(scores))


# 11. SCIPY HYPOTHESIS TESTING
def scipy_analysis(df):
    fake_scores = df.loc[
        df["actual_label"] == "FAKE", "score"
    ].to_numpy(dtype=float)

    real_scores = df.loc[
        df["actual_label"] == "REAL", "score"
    ].to_numpy(dtype=float)

    print("\n========== SCIPY HYPOTHESIS TESTING ==========")

    if len(fake_scores) < 2 or len(real_scores) < 2:
        print("Not enough data for t-test.")
        return

    print("Fake group size:", len(fake_scores))
    print("Real group size:", len(real_scores))
    print("Mean fake score:", np.mean(fake_scores))
    print("Mean real score:", np.mean(real_scores))

    # H0: Population mean scores are equal.
    # H1: Population mean scores differ.
    t_value, p_value = stats.ttest_ind(
        fake_scores,
        real_scores,
        equal_var=False,
        nan_policy="omit"
    )

    print("T-value:", round(float(t_value), 4))
    print("P-value:", float(p_value))

    alpha = 0.05

    if p_value < alpha:
        print("Reject H0: Significant difference in mean scores.")
    else:
        print("Fail to reject H0: Insufficient evidence of a difference.")


# 12. SAVE RESULTS
def save_results(df):
    try:
        df.to_csv(OUTPUT_FILE, index=False)
        print("Results saved to:", OUTPUT_FILE)
    except Exception as e:
        print("Error saving results:", e)


# 13. MAIN MENU
def main():
    df = load_data()

    if df is None:
        print("Program stopped. Check your uploaded files.")
        return

    df = process_data(df)

    while True:
        print("\n====== FAKE NEWS DETECTION SYSTEM ======")
        print("1. Display news records")
        print("2. Check a new headline")
        print("3. Add a headline")
        print("4. Update a headline")
        print("5. Delete a headline")
        print("6. NumPy statistical analysis")
        print("7. SciPy hypothesis testing")
        print("8. Save results to CSV")
        print("9. Exit")

        choice = input("Enter your choice (1-9): ").strip()

        if choice == "1":
            display_news(df)

        elif choice == "2":
            title = input("Enter headline: ").strip()

            if title:
                print("Suspicion score:", calculate_score(title))
                print("Prediction:", detect_news(title))
                print("Note: This is a keyword-based prediction.")
            else:
                print("Headline cannot be empty.")

        elif choice == "3":
            df = add_news(df)

        elif choice == "4":
            df = update_news(df)

        elif choice == "5":
            df = delete_news(df)

        elif choice == "6":
            numpy_analysis(df)

        elif choice == "7":
            scipy_analysis(df)

        elif choice == "8":
            save_results(df)

        elif choice == "9":
            save_results(df)
            print("Program completed.")
            break

        else:
            print("Invalid choice. Enter a number from 1 to 9.")


main()
