import sys
import requests
import time

def get_user_location(user_url):
    response = requests.get(user_url)

    if response.status_code == 404:
        print("Not found")
        return
    elif response.status_code == 403:
        reset_time = int(response.headers['X-Ratelimit-Reset'])
        current_time = int(time.time())
        reset_in_minutes = (reset_time - current_time) // 60
        print(f"Reset in {reset_in_minutes} min")
        return

    user_data = response.json()
    location = user_data.get('location', 'Location not provided')
    print(location)