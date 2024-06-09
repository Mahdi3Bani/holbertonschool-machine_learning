#!/usr/bin/env python3
"""displays the number of launches per rocket"""
import requests
from collections import defaultdict


def rocket_frequency():
    """displays the number of launches per rocket"""
    rockets = defaultdict(int)
    launches = requests.get('https://api.spacexdata.com/v4/launches').json()

    for launch in launches:
        rocket_id = launch.get('rocket')
        rocket = requests.\
            get('https://api.spacexdata.com/v4/rockets/{}'
                .format(rocket_id)).json()
        rocket_name = rocket.get('name')
        rockets[rocket_name] += 1

    sorted_rockets = sorted(
        rockets.items(), key=lambda kv: kv[1], reverse=True)

    for rocket_name, count in sorted_rockets:
        print('{}: {}'.format(rocket_name, count))

if __name__ == '__main__':
    rocket_frequency()
