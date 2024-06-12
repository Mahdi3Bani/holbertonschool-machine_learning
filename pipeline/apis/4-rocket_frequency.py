#!/usr/bin/env python3
"""api module"""
import requests
from collections import defaultdict

def rocket_frequency():
    """Get the number of launches for each rocket."""
    rockets = defaultdict(int)
    launches = requests.get('https://api.spacexdata.com/v4/launches').json()

    for launch in launches:
        rocket_id = launch.get('rocket')
        rocket = requests.get(f'https://api.spacexdata.com/v4/rockets/{rocket_id}').json()
        rocket_name = rocket.get('name')
        rockets[rocket_name] += 1

    sorted_rockets = sorted(rockets.items(), key=lambda kv: kv[1], reverse=True)
    return sorted_rockets

if __name__ == '__main__':
    rocket_launch_counts = rocket_frequency()
    for rocket_name, count in rocket_launch_counts:
        print(f'{rocket_name}: {count}')
