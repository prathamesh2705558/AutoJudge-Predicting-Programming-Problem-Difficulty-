import requests
from bs4 import BeautifulSoup

def test_scrape(url):
    print(f"Testing URL: {url}")
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            # Title
            title = soup.find('div', class_='title')
            print(f"Title: {title.text.strip() if title else 'Not Found'}")
            
            # Problem Statement (Codeforces usually uses class 'problem-statement')
            statement = soup.find('div', class_='problem-statement')
            if statement:
                print("Problem Statement Found!")
                # Get text length
                text = statement.get_text()
                print(f"Text Length: {len(text)}")
                print(f"Sample: {text[:200]}...")
            else:
                print("Problem Statement NOT found (might be different structure or blocked)")
        else:
            print("Failed to fetch.")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Test with a standard Codeforces problem (Watermelon)
    test_scrape("https://codeforces.com/problemset/problem/4/A")
