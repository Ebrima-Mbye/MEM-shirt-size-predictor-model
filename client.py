"""Example client to query the deployed shirt-size model."""

import requests


url = "http://0.0.0.0:8000/predict"

examples = [
    {
        "height_cm": 165,
        "weight_kg": 58,
        "age": 22,
        "gender": "female",
        "fit_preference": "regular",
        "build": "lean",
    },
    {
        "height_cm": 178,
        "weight_kg": 82,
        "age": 28,
        "gender": "male",
        "fit_preference": "regular",
        "build": "athletic",
    },
    {
        "height_cm": 172,
        "weight_kg": 95,
        "age": 35,
        "gender": "male",
        "fit_preference": "oversized",
        "build": "average",
    },
]


for payload in examples:
    r = requests.post(url, json=payload, timeout=10)
    r.raise_for_status()
    print(payload, "=>", r.json())


