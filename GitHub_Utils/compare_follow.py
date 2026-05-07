import json
import os

followers_dir = "followers"
following_dir = "following"
os.makedirs(followers_dir, exist_ok=True)
os.makedirs(following_dir, exist_ok=True)


def compare_followers(old_file, new_file):
    with open(old_file, "r") as f:
        old_followers = set(json.load(f))
    
    with open(new_file, "r") as f:
        new_followers = set(json.load(f))
    
    new_followers_added = new_followers - old_followers
    followers_removed = old_followers - new_followers
    
    print("New Added:", len(new_followers_added))
    print("New Added List:", [f for f in new_followers_added])
    print("_" * 40)
    print("Removed:", len(followers_removed))
    print("Removed List:", [f for f in followers_removed])

if __name__ == "__main__":
    old_file = followers_dir + "/" + "followers2026-05-06.json"  # Change this to the previous followers or following file
    new_file = followers_dir + "/" + "followers2026-05-07.json"  # Change this to the new followers or following file

    compare_followers(old_file, new_file)