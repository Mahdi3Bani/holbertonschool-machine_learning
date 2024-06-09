#!/usr/bin/env python3
"""Get location of github user"""
import requests
import sys
import time


def user_location(url):
    """Get location of github user"""
    req = requests.get(url)
    if req.status_code == 403:
        reset = int(req.headers['X-Ratelimit-Reset'])
        tm_to_reset = reset - int(time.time())
        print("Reset in {} min".format(tm_to_reset//60))
    elif req.status_code == 404:
        print("Not found")
    else:
        user = req.json()
        print(user['location'])


if __name__ == "__main__":
    user_location(sys.argv[1])
