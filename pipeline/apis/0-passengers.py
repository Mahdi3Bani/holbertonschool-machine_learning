#!/usr/bin/env python3
"""passenfers"""
import requests
import json

def availableShips(passengerCount):
    """availableShips"""
    ships = []
    url = "https://swapi-api.hbtn.io/api/starships"
    
    while url:
        response = requests.get(url)
        data = response.json()
        
        for ship in data['results']:
            passengers = ship["passengers"]
            if passengers not in ["n/a", "unknown"]:
                passengers = passengers.replace(',', '')    
                
                if int(passengers) >= passengerCount:
                    ships.append(ship['name'])
        
        url = data['next']
    
    return ships