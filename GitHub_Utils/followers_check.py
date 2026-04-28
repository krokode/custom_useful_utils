import requests
import json
import datetime
from dotenv import load_dotenv
import os

load_dotenv()


username = os.getenv("GITHUB_USERNAME")

followers = []
page = 1

followers_dir = "followers"
os.makedirs(followers_dir, exist_ok=True)

def get_followers(username, page):
    url = f"https://api.github.com/users/{username}/followers?per_page=100&page={page}"
    response = requests.get(url).json()
    return response

if __name__ == "__main__":
    while True:
        response = get_followers(username, page)
    
        if not response:
            break
        
        followers.extend(response)
        page += 1

    usernames = sorted([f["login"] for f in followers])
    print("Number of Followers:", len(usernames))

    d = datetime.datetime.now().strftime("%Y-%m-%d")
    file_name = f"followers{d}.json"

    with open(followers_dir + "/" + file_name, "w") as f:
        json.dump(usernames, f)

    print(f"Followers saved to {followers_dir}/{file_name}")

