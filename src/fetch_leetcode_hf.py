from datasets import load_dataset
import pandas as pd
import numpy as np
import re

def clean_html(raw_html):
    """Remove HTML tags from description."""
    if not isinstance(raw_html, str):
        return ""
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext

def fetch_leetcode():
    print("Attempting to load 'newfacade/LeetCodeDataset'...")
    try:
        # Load the dataset
        dataset = load_dataset("newfacade/LeetCodeDataset", split="train", streaming=True)
        
        print("Dataset stream opened. Fetching rows...")
        new_data = []
        
        # We'll fetch up to 10000 LeetCode problems (likely captures all)
        for i, item in enumerate(dataset):
            if i >= 10000:
                break
                
            # Inspect first item to know keys
            if i == 0:
                print("LeetCode Keys:", item.keys())
            
            # Extract fields
            # Based on common schemas for this dataset
            # content: description
            # difficulty: Difficulty level
            
            difficulty = item.get('difficulty', 'Medium')
            
            # Assign synthetic score based on difficulty since LeetCode has no numeric rating usually
            if difficulty == 'Easy':
                score = np.random.randint(800, 1200)
            elif difficulty == 'Medium':
                score = np.random.randint(1300, 1800)
            elif difficulty == 'Hard':
                score = np.random.randint(1900, 2500)
            else:
                score = 1500
            
            row = {
                'title': item.get('title', f'LeetCode {i}'),
                'description': clean_html(item.get('content', '')), # Clean HTML if present
                'input_description': '', # LeetCode descriptions often mix these
                'output_description': '',
                'problem_score': score,
                'problem_class': difficulty
            }
            new_data.append(row)
            
        print(f"Fetched {len(new_data)} LeetCode problems.")
        
        # Load existing data
        try:
            existing_df = pd.read_csv("data/dataset.csv")
            print(f"Existing dataset size: {len(existing_df)}")
        except FileNotFoundError:
            existing_df = pd.DataFrame()
            
        # Combine
        new_df = pd.DataFrame(new_data)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        
        # Save
        combined_df.to_csv("data/dataset.csv", index=False)
        print(f"Updated dataset size: {len(combined_df)}")
        print("Saved merged data to data/dataset.csv")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error loading: {e}")

if __name__ == "__main__":
    fetch_leetcode()
