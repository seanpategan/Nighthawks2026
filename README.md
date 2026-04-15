# Brooklyn Bridge Traffic Monitor

Real-time vehicle detection on the Brooklyn Bridge using public NYC DOT camera feeds.

## What it does
Pulls live camera frames every 10 seconds, runs YOLOv8 to count cars, trucks, buses, and motorcycles, and displays a live dashboard with counts and a density score (1–10) for both inbound and outbound directions.

## Setup
```
pip install -r requirements.txt
python camera.py
```
Then serve the frontend:
```
python -m http.server 8080
```
Open `http://localhost:8080`.

## Density score
Relative to the last 20 frames. 1 = lightest traffic seen recently, 10 = heaviest.

## Swap cameras
Change the two URLs in `camera.py` to point at any public camera feed.
