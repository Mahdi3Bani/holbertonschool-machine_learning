#!/usr/bin/env python3
"""api module"""
import requests

def first_launch():
    """Get details of the upcoming SpaceX launch."""
    launches = requests.get('https://api.spacexdata.com/v4/launches/upcoming').json()
    unix_dates = [launch['date_unix'] for launch in launches]
    min_idx = unix_dates.index(min(unix_dates))
    upcoming_launch = launches[min_idx]

    rocket = requests.get(f'https://api.spacexdata.com/v4/rockets/{upcoming_launch["rocket"]}').json()
    launchpad = requests.get(f'https://api.spacexdata.com/v4/launchpads/{upcoming_launch["launchpad"]}').json()

    return {
        'launch_name': upcoming_launch['name'],
        'launch_date': upcoming_launch['date_local'],
        'rocket_name': rocket['name'],
        'launchpad_name': launchpad['name'],
        'launchpad_locality': launchpad['locality']
    }

if __name__ == '__main__':
    details = first_launch()
    print(f"{details['launch_name']} ({details['launch_date']}) {details['rocket_name']} - {details['launchpad_name']} ({details['launchpad_locality']})")
