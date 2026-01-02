import pandas as pd

def relabel_dataset():
    input_path = "data/dataset.csv"
    print(f"Loading {input_path}...")
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print("Dataset not found!")
        return

    print("Original Distribution:")
    print(df['problem_class'].value_counts(normalize=True))

    def assign_class(row):
        # Trust existing LeetCode classes (Easy/Medium/Hard) if score is synthetic
        # But for Codeforces (real ratings), re-bucket them.
        try:
            score = float(row['problem_score'])
            
            # New Thresholds to increase "Medium" bucket
            if score < 1300:
                return 'Easy'
            elif score < 1900:
                return 'Medium'
            else:
                return 'Hard'
        except:
            # If score is invalid, keep original class or default to Medium
            return row['problem_class']

    # Apply new logic
    df['problem_class'] = df.apply(assign_class, axis=1)

    print("\nNew Distribution:")
    print(df['problem_class'].value_counts(normalize=True))

    df.to_csv(input_path, index=False)
    print(f"\nSaved re-labeled data to {input_path}")

if __name__ == "__main__":
    relabel_dataset()
