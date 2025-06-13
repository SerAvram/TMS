import json, os

# Шукаємо папку data поруч із logic
DATA_DIR = os.path.join(os.path.dirname(__file__), os.pardir, 'data')

def load_data():
    # вантажимо список транспортних засобів
    with open(os.path.join(DATA_DIR, 'transport.json'), encoding='utf-8') as f:
        transports = json.load(f)
    # вантажимо координати зупинок
    with open(os.path.join(DATA_DIR, 'stops.json'), encoding='utf-8') as f:
        stops = json.load(f)

    # перетворюємо route = ["A","B","C"] → coords = [ {lat,lon}, … ]
    for tr in transports:
        tr['coords'] = [stops[name] for name in tr['route']]
    return transports
