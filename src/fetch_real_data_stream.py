from datasets import load_dataset
import pandas as pd

def fetch_data():
    print("Attempting to stream 'open-r1/codeforces' dataset...")
    try:
        # Use streaming=True to avoid downloading the whole dataset
        dataset = load_dataset("open-r1/codeforces", split="train", streaming=True)
        
        print("Dataset stream opened. Fetching first 15000 rows...")
        data_list = []
        for i, item in enumerate(dataset):
            if i >= 15000:
                break
            
            # Extract relevant fields. 
            # Note: The schema might vary. Let's inspect the first item.
            if i == 0:
                print("First item keys:", item.keys())

            # Map to our expected schema if possible
            # 'problem' usually contains the description. 
            # 'rating' is the difficulty score.
            # 'difficulty' might be text.
            
            row = {
                'title': item.get('title', f'Problem {i}'),
                'description': item.get('description', ''),
                'input_description': item.get('input_format', ''),
                'output_description': item.get('output_format', ''),
                'problem_score': item.get('rating', 0),
                'problem_class': 'Medium' # Placeholder, will calculate later
            }
            
            # Simple heuristic for class if not present
            try:
                score = int(row['problem_score'])
                if score < 1200: row['problem_class'] = 'Easy'
                elif score < 1600: row['problem_class'] = 'Medium'
                else: row['problem_class'] = 'Hard'
            except:
                pass
                
            data_list.append(row)
            
        print(f"Fetched {len(data_list)} rows.")
        
        # Convert to DataFrame
        df = pd.DataFrame(data_list)
        
        # Save to CSV
        output_path = "data/dataset.csv"
        df.to_csv(output_path, index=False)
        print(f"Saved real data to {output_path}")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error loading: {e}")

if __name__ == "__main__":
    fetch_data()
