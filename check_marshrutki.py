import requests

# Ми бачимо, що в сервисі transport_stage:
#   id=6  → "Маршрутне таксі"  (geometryType = esriGeometryPolyline)
#
FEATURESERVER_BASE = (
    "https://gisserver-stage.kyivcity.gov.ua/"
    "mayno/rest/services/KYIV_API/транспорт_stage/FeatureServer"
)
MARSHRUTKA_LAYER_ID = 6  # саме цей ID відповідає за «Маршрутне таксі» у транспорт_stage
MARSHRUTKA_LAYER_URL = f"{FEATURESERVER_BASE}/{MARSHRUTKA_LAYER_ID}/query"

GEOCODE_HEADERS = {
    "User-Agent": "MyTransitApp/Debug (dump_marshrutki)"
}

def dump_all_marshrutka_features():
    params = {
        "where": "1=1",
        "outFields": "from_code1,to_code1,num_route",
        "returnGeometry": True,
        "f": "json"    # ⇐ ArcGIS JSON (не 'geojson')
    }

    try:
        resp = requests.get(
            MARSHRUTKA_LAYER_URL,
            params=params,
            headers=GEOCODE_HEADERS,
            timeout=10
        )
        resp.raise_for_status()
    except Exception as e:
        print("❌ Помилка під час запиту до шару 6 (Маршрутне таксі):", e)
        return

    data = resp.json()

    # Якщо ArcGIS повернув помилку всередині JSON
    if "error" in data:
        print("❌ ArcGIS‐помилка:", data["error"])
        return

    features = data.get("features", [])
    if not features:
        print("⚠ Шар 6 повернув 0 фіч. Можливо, зараз (у transport_stage) неактивні маршрутки.")
        return

    print(f"🔹 Знайдено {len(features)} фіч у шарі 6 (Маршрутне таксі, transport_stage):\n")
    for i, feat in enumerate(features, start=1):
        # У ArcGIS JSON усі атрибути лежать у feat['attributes'], а не у 'properties'
        attrs = feat.get("attributes", {})
        a = attrs.get("from_code1", "")
        b = attrs.get("to_code1", "")
        raw_route = str(attrs.get("num_route", "") or "")
        route = raw_route.split("_")[0]

        # Геометрія лежить у feat['geometry']
        geom = feat.get("geometry", {})
        paths = geom.get("paths", None)
        coords = geom.get("coordinates", None)

        if paths and isinstance(paths, list) and len(paths) > 0:
            total_points = sum(len(path) for path in paths)
            geom_desc = f"paths (множина ліній, {len(paths)} частин, {total_points} точок)"
        elif coords and isinstance(coords, list):
            # coords може бути або [[lon,lat],…] (single line), або [[[lon,lat],…], …] (multiline)
            if isinstance(coords[0], list) and isinstance(coords[0][0], list):
                total_points = sum(len(line) for line in coords)
                geom_desc = f"coordinates (множина ліній, {len(coords)} частин, {total_points} точок)"
            else:
                total_points = len(coords)
                geom_desc = f"coordinates (single line, {total_points} точок)"
        else:
            geom_desc = "без геометрії"

        print(f"#{i}: маршрут {route}, зуп. {a} → {b}, геометрія: {geom_desc}")

if __name__ == "__main__":
    dump_all_marshrutka_features()
