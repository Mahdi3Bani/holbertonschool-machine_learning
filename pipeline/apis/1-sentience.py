#!/usr/bin/env python3
"""1-sentience"""
import requests
import json


def sentientPlanets():
    """sentientPlanets"""
    planets = []
    url = "https://swapi-api.hbtn.io/api/species"
    
    while url:
        response = requests.get(url)
        data = response.json()
        
        for species in data["results"]:
            classification = species["classification"]
            designation = species["designation"]
            if classification in ["sentient"] or designation in ["sentient"]:
                url2 = species["homeworld"]
                if url2 is not None:
                    response2 = requests.get(url2)
                    data2 = response2.json()
                    planets.append(data2["name"])

        
        url = data['next']
    
    return planets