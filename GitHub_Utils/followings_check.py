import requests
import json
import datetime
from dotenv import load_dotenv
import os

load_dotenv()


username = os.getenv("GITHUB_USERNAME")

following = []
page = 1

following_dir = "following"
os.makedirs(following_dir, exist_ok=True)

def get_following(username, page):
    url = f"https://api.github.com/users/{username}/following?per_page=100&page={page}"
    response = requests.get(url).json()
    return response

if __name__ == "__main__":
    while True:
        response = get_following(username, page)
        
        if not response:
            break
        
        following.extend(response)
        page += 1

    usernames = sorted([f["login"] for f in following])
    print("Number of Following:", len(usernames))

    d = datetime.datetime.now().strftime("%Y-%m-%d")
    file_name = f"following{d}.json"

    with open(following_dir + "/" + file_name, "w") as f:
        json.dump(usernames, f)

    print(f"Following saved to {following_dir}/{file_name}")

