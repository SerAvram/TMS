from kivy.graphics import Color, Ellipse, Line, Rectangle
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.boxlayout import BoxLayout
from kivy.core.text import Label as CoreLabel

import networkx as nx
from kivy.config import Config
Config.set('input', 'mouse', 'mouse,disable_multitouch')
Config.set('modules', 'touchring', '0')

import json, heapq, requests, math
from kivy.app            import App
from kivy.lang           import Builder
from kivy.uix.boxlayout  import BoxLayout
from kivy.properties     import ObjectProperty
from kivy.clock          import Clock, mainthread
from kivy_garden.mapview import MapView, MapMarker, MapLayer
from google.transit      import gtfs_realtime_pb2

KV = '''
<RootWidget>:
    orientation: 'vertical'
    mapview: mapview

    BoxLayout:
        size_hint_y: None
        height: '40dp'
        padding: '4dp'
        spacing: '4dp'

        TextInput:
            id: addrA
            hint_text: 'Адреса А'
            multiline: False
        TextInput:
            id: addrB
            hint_text: 'Адреса Б'
            multiline: False
        Button:
            text: 'Прокласти маршрут'
            size_hint_x: None
            width: '140dp'
            on_release: root.on_route(addrA.text, addrB.text)

    MapView:
        id: mapview
        lat: 50.45
        lon: 30.523
        zoom: 12
'''

# ————— Налаштування джерел —————
FEED_URL = "http://193.23.225.214:732/api/realtime"
FEATURESERVER_BASE = (
    "https://gisserver-stage.kyivcity.gov.ua/"
    "mayno/rest/services/KYIV_API/транспорт_stage/FeatureServer"
)
FETCH_INTERVAL = 5.0                # інтервал оновлення RT
CLIP_THRESHOLD_METERS = 20          # обрізка сегментів між зупинками
STOPS_LAYER  = FEATURESERVER_BASE + "/0/query"
ROUTE_LAYERS = [6, 2, 3, 4]         # 6=маршрутка, 2=автобус, 3=тролейбус, 4=трамвай

GEOCODE_HEADERS = {"User-Agent": "MyTransitApp/1.0 (contact@yourdomain.com)"}

BUS_COLOR        = (1, 0.5, 0, 1)
TROLLEY_COLOR    = (0, 0.5, 1, 1)
TRAM_COLOR       = (0, 1, 0, 1)
MARSHRUTKA_COLOR = (1, 1, 0, 1)

LAYER_MAP = {
    "маршрутка": 6, "marshrutka": 6,
    "автобус":   2, "bus": 2,
    "тролейбус": 3, "trolleybus": 3,
    "трамвай":   4, "tram": 4
}

# ————— Зчитуємо vehicles.json лише раз, формуємо static_map —————
with open("vehicles.json", encoding="utf-8") as f:
    static_data = json.load(f)

static_map = {}
for v in static_data:
    raw_rid = v.get("routeId", "")
    if not raw_rid:
        continue
    raw_rid = str(raw_rid).strip().lower()
    bt = v.get("bodyType", "").lower().strip()
    if not bt:
        continue

    uid = str(v.get("uid", "")).strip()
    if not uid:
        continue

    for rid_part in raw_rid.split(","):
        rid = rid_part.strip()
        if not rid:
            continue

        if rid not in static_map:
            static_map[rid] = []

        duplicate = False
        for existing_rec in static_map[rid]:
            if existing_rec["uid"] == uid:
                duplicate = True
                break
        if duplicate:
            continue

        static_map[rid].append({
            "uid":      uid,
            "bodyType": bt,
            "status":   v.get("status", "").lower().strip()
        })

# ————— Кеш «сирих» сегментів для fetch_route_shape —————
shape_cache = {}

# ————— Додатковий локальний кеш злитих polyline для повторних викликів —————\nmerged_cache = {}


def fetch_route_shape(route_id, forced_bodyType=None):
    if forced_bodyType:
        bt = forced_bodyType
    else:
        recs = static_map.get(route_id, [])
        if recs:
            bt = recs[0]["bodyType"]
        else:
            bt = None

    tried_layers = []
    if bt and bt in LAYER_MAP:
        tried_layers.append(LAYER_MAP[bt])
    for lid in ROUTE_LAYERS:
        if lid not in tried_layers:
            tried_layers.append(lid)

    for layer_id in tried_layers:
        key = f"{layer_id}:{route_id}"
        if key in shape_cache:
            return shape_cache[key]

        url = f"{FEATURESERVER_BASE}/{layer_id}/query"
        params = {
            "where": f"num_route = '{route_id}'",
            "outFields": "*",
            "returnGeometry": True,
            "f": "geojson"
        }
        try:
            resp = requests.get(url, params=params, timeout=5)
            resp.raise_for_status()
            data = resp.json()
        except:
            continue

        feats = data.get("features", [])
        if not feats:
            shape_cache[key] = []
            continue

        segments = []
        for feat in feats:
            geom = feat.get("geometry", {})
            coords = geom.get("paths", None) or geom.get("coordinates", [])
            if not coords:
                continue
            if isinstance(coords[0], list) and isinstance(coords[0][0], list):
                for line in coords:
                    seg = [(lat, lon) for lon, lat in line]
                    segments.append(seg)
            else:
                seg = [(lat, lon) for lon, lat in coords]
                segments.append(seg)

        if segments:
            shape_cache[key] = segments
            return segments

    if tried_layers:
        shape_cache[f"{tried_layers[0]}:{route_id}"] = []
    return []


def merge_segments(segments):
    if not segments:
        return []
    used = [False]*len(segments)
    merged = segments[0][:]
    used[0] = True

    def point_dist(a,b):
        return ((a[0]-b[0])**2 + (a[1]-b[1])**2)**0.5

    changed = True
    while changed:
        changed = False
        best_idx = None
        best_swap = False
        best_dist = float('inf')
        tail_pt = merged[-1]
        for i, seg in enumerate(segments):
            if used[i]:
                continue
            d0 = point_dist(tail_pt, seg[0])
            d1 = point_dist(tail_pt, seg[-1])
            if d0 < best_dist:
                best_dist = d0
                best_idx = i
                best_swap = False
            if d1 < best_dist:
                best_dist = d1
                best_idx = i
                best_swap = True
        if best_idx is not None:
            seg = segments[best_idx][:]
            if best_swap:
                seg.reverse()
            if abs(merged[-1][0]-seg[0][0])<1e-7 and abs(merged[-1][1]-seg[0][1])<1e-7:
                merged.extend(seg[1:])
            else:
                merged.extend(seg)
            used[best_idx] = True
            changed = True

    changed = True
    while changed:
        changed = False
        best_idx = None
        best_swap = False
        best_dist = float('inf')
        head_pt = merged[0]
        for i, seg in enumerate(segments):
            if used[i]:
                continue
            d0 = point_dist(head_pt, seg[0])
            d1 = point_dist(head_pt, seg[-1])
            if d0 < best_dist:
                best_dist = d0
                best_idx = i
                best_swap = True
            if d1 < best_dist:
                best_dist = d1
                best_idx = i
                best_swap = False
        if best_idx is not None:
            seg = segments[best_idx][:]
            if best_swap:
                seg.reverse()
            if abs(merged[0][0]-seg[-1][0])<1e-7 and abs(merged[0][1]-seg[-1][1])<1e-7:
                merged = seg[:-1] + merged
            else:
                merged = seg + merged
            used[best_idx] = True
            changed = True

    return merged


def find_closest_index(polyline, target_latlon):
    min_d = float('inf')
    min_i = None
    tlat, tlon = target_latlon
    for i, (plat, plon) in enumerate(polyline):
        d = (plat - tlat)**2 + (plon - tlon)**2
        if d < min_d:
            min_d = d
            min_i = i
    return min_i


def crop_single_segment_between_stops(polys, oid_board, oid_alight, stops):
    """
    Приймає:
      polys        — список polyline (кожен елемент — список точок [(lat, lon), …])
      oid_board    — код boarding‐зупинки
      oid_alight   — код alighting‐зупинки
      stops        — словник {stop_id: (lat, lon, name)}

    Повертає:
      список списків точок [[(lat1, lon1), (lat2, lon2), …],  … ] (можливо, один елемент),
      або [] якщо обрізати не вдається.
    """

    # ────────────────────────────────────────────────────────────────────────────────
    # Якщо polys пустий або polys[0] пустий, повертаємо [] негайно.
    if not polys or not polys[0]:
        return []
    # ────────────────────────────────────────────────────────────────────────────────

    # 1) Беремо перший polyline (shape)
    poly = polys[0]

    # 2) Координати центру boarding та alighting
    la_board, lo_board, _ = stops[oid_board]
    la_alight, lo_alight, _ = stops[oid_alight]

    # 3) Знаходимо індекс bi (найближча точка полігону до boarding)
    dist_to_shape = [
        math.sqrt((la_board - p_la)**2 + (lo_board - p_lo)**2) * 111000
        for (p_la, p_lo) in poly
    ]
    bi_candidate = min(range(len(dist_to_shape)), key=lambda i: dist_to_shape[i])
    if dist_to_shape[bi_candidate] > CLIP_THRESHOLD_METERS:
        return []
    bi = bi_candidate

    # 4) Знаходимо індекс ai (найближча точка полігону до alighting)
    dist_to_shape_alight = [
        math.sqrt((la_alight - p_la)**2 + (lo_alight - p_lo)**2) * 111000
        for (p_la, p_lo) in poly
    ]
    ai_candidate = min(range(len(dist_to_shape_alight)), key=lambda i: dist_to_shape_alight[i])
    if dist_to_shape_alight[ai_candidate] > CLIP_THRESHOLD_METERS:
        return []
    ai = ai_candidate

    # 5) Обрізаємо (crop) shape за індексами
    if bi <= ai:
        segment = poly[bi : ai + 1]
    else:
        segment = list(reversed(poly[ai : bi + 1]))

    # 6) Спрощена перевірка напрямку segment (достатньо перевірити тільки першу точку)
    if segment:
        first_pt = segment[0]
        dist_first_to_board  = math.sqrt((first_pt[0] - la_board)**2  + (first_pt[1] - lo_board)**2) * 111000
        dist_first_to_alight = math.sqrt((first_pt[0] - la_alight)**2 + (first_pt[1] - lo_alight)**2) * 111000
        if dist_first_to_alight < dist_first_to_board:
            segment.reverse()

    cropped = [segment]

    # 7) Якщо segment пустий, робимо fallback до двох найближчих точок самої poly
    if not segment:
        best_pt_board = min(
            poly,
            key=lambda p: math.sqrt((p[0] - la_board)**2 + (p[1] - lo_board)**2) * 111000
        )
        best_pt_alight = min(
            poly,
            key=lambda p: math.sqrt((p[0] - la_alight)**2 + (p[1] - lo_alight)**2) * 111000
        )
        cropped = [[best_pt_board, best_pt_alight]]

    return cropped


class StopMarker(MapMarker):
    def __init__(self, lat, lon, text, color, **kwargs):
        super().__init__(lat=lat, lon=lon, size=(24,24), **kwargs)
        with self.canvas.before:
            Color(*color)
            self._circle = Ellipse(size=self.size, pos=self.pos)
        self._lbl = CoreLabel(text=str(text), font_size=12)
        self._lbl.refresh()
        with self.canvas:
            Color(1,1,1,1)
            self._tex = Rectangle(texture=self._lbl.texture,
                                  size=self._lbl.texture.size,
                                  pos=self.pos)
        self.bind(pos=self._update)

    def _update(self, *args):
        self._circle.pos = (self.x - self.width/2, self.y - self.height/2)
        tw, th = self._lbl.texture.size
        self._tex.pos = (self.x - tw/2, self.y - th/2)


class RouteLayer(MapLayer):
    def __init__(self, segments, color, **kwargs):
        super().__init__(**kwargs)
        self.segments = segments
        self.color    = color

    def reposition(self, *args):
        # ---------------------------------------------------
        # Якщо батьківського MapView у нас нема, просто нічого не робимо
        if self.parent is None:
            return
        # ---------------------------------------------------
        self.canvas.clear()
        if not self.segments:
            return
        with self.canvas:
            Color(*self.color)
            for seg in self.segments:
                pts = []
                for lat, lon in seg:
                    # Тепер self.parent гарантовано не None → викликаємо get_window_xy_from
                    x, y = self.parent.get_window_xy_from(lat, lon, self.parent.zoom)
                    pts.extend([x, y])
                Line(points=pts, width=3)



class VehicleMarker(MapMarker):
    def __init__(self, raw_id, route_id, bodyType, lat, lon, **kw):
        super().__init__(lat=lat, lon=lon, size=(32,32), **kw)
        c = BUS_COLOR
        if bodyType == "тролейбус": c = TROLLEY_COLOR
        if bodyType == "трамвай":   c = TRAM_COLOR
        if bodyType == "маршрутка": c = MARSHRUTKA_COLOR
        with self.canvas.before:
            Color(*c)
            self._circle = Ellipse(size=self.size, pos=self.pos)

        self._lbl = CoreLabel(text=str(route_id), font_size=12)
        self._lbl.refresh()
        with self.canvas:
            Color(1, 1, 1, 1)
            self._tex = Rectangle(texture=self._lbl.texture,
                                  size=self._lbl.texture.size,
                                  pos=self.pos)

        self.route_id = route_id
        self.bodyType = bodyType
        self.bind(pos=self._update)

    def _update(self, *a):
        self._circle.pos = (self.x - self.width/2, self.y - self.height/2)
        tw, th = self._lbl.texture.size
        self._tex.pos = (self.x - tw/2, self.y - th/2)

    def on_touch_down(self, touch):
        if 'button' in touch.profile and touch.button != 'left':
            return super().on_touch_down(touch)
        if self.collide_point(*touch.pos):
            print(f"[CLICK] route_id={self.route_id}, bodyType={self.bodyType}")
            App.get_running_app().root.show_route(self.route_id, self.bodyType)
            return True
        return super().on_touch_down(touch)


class RealtimeLayer:
    def __init__(self, mapview):
        self.mapview = mapview
        self.vehicles = {}
        # Перший виклик fetch_data через 0.5 с — надалі цей же метод сам плануватиме повторні виклики
        Clock.schedule_once(self.fetch_data, 0.5)

    def fetch_data(self, dt):
        try:
            r = requests.get(FEED_URL, timeout=5)
            feed = gtfs_realtime_pb2.FeedMessage()
            feed.ParseFromString(r.content)
        except:
            # Якщо запит провалився — просто плануємо ще одну спробу через FETCH_INTERVAL
            Clock.schedule_once(self.fetch_data, FETCH_INTERVAL)
            return

        seen = set()
        for ent in feed.entity:
            if not ent.HasField("vehicle"):
                continue
            veh = ent.vehicle
            if not veh.HasField("trip"):
                continue
            trip = veh.trip

            fid = veh.vehicle.id or str(ent.id)
            raw_label = str(veh.vehicle.label or "").strip().lower()
            trip_lane = str(trip.route_id or "").strip().lower()

            parts = raw_label.split(",")
            first_chunk = parts[0].strip()

            if first_chunk in static_map:
                candidates = static_map[first_chunk]
            else:
                if first_chunk.endswith("т") and first_chunk[:-1] in static_map:
                    candidates = static_map[first_chunk[:-1]]
                    first_chunk = first_chunk[:-1]
                else:
                    continue

            route_id = first_chunk

            # Шукаємо match по first character від trip.route_id → обираємо правильний bodyType
            chosen_rec = None
            if trip_lane:
                first_trip_char = trip_lane[0]
                for rec in candidates:
                    if rec["uid"] and rec["uid"][0].lower() == first_trip_char:
                        chosen_rec = rec
                        break
            if not chosen_rec:
                chosen_rec = candidates[0]

            chosen_bt = chosen_rec["bodyType"]
            if not chosen_bt:
                continue
            # не малюємо маршрутки
            if chosen_bt in ("маршрутка", "marshrutka"):
                continue

            lat, lon = veh.position.latitude, veh.position.longitude

            if fid in self.vehicles:
                m = self.vehicles[fid]
                m.lat, m.lon = lat, lon
            else:
                m = VehicleMarker(fid, route_id, chosen_bt, lat, lon)
                self.vehicles[fid] = m
                self.mapview.add_marker(m)

            seen.add(fid)

        for fid in list(self.vehicles):
            if fid not in seen:
                self.mapview.remove_marker(self.vehicles.pop(fid))

        # Після кожного виконання fetch_data плануємо наступний виклик через FETCH_INTERVAL секунд.
        Clock.schedule_once(self.fetch_data, FETCH_INTERVAL)



class RootWidget(BoxLayout):
    mapview = ObjectProperty()

    def __init__(self, **kw):
        super().__init__(**kw)

        # 1) Завантажуємо зупинки:
        self.stops = self._fetch_stops()
        print(f"[DEBUG] Loaded {len(self.stops)} stops")

        # 2) Будуємо граф маршрутів:
        self.graph = self._build_graph()
        total_edges = sum(len(edges) for edges in self.graph.values())
        print(f"[DEBUG] Built graph with {len(self.graph)} nodes and {total_edges} edges")

        # 3) Формуємо stops_by_route і routes_by_stop:
        from collections import defaultdict
        tmp = defaultdict(set)
        for u, neighbors in self.graph.items():
            for v, edges in neighbors.items():
                for length, rt in edges:
                    if rt != 'walk':
                        tmp[rt].add(u)
                        tmp[rt].add(v)
        self.stops_by_route = {rt: list(stops) for rt, stops in tmp.items()}
        self.routes_by_stop = {}
        for rt, stops_list in self.stops_by_route.items():
            for sid in stops_list:
                self.routes_by_stop.setdefault(sid, []).append(rt)

        # 4) Прибираємо зайвий schedule_interval, лишаємо тільки цей об’єкт:
        self.rt = RealtimeLayer(self.mapview)
        # <--- Ось тут більше НЕ викликаємо:
        # Clock.schedule_interval(self.rt.fetch_data, FETCH_INTERVAL)

        self.stop_markers = []


    def build_transport_graph(self):
        G = nx.MultiDiGraph()
        G.add_nodes_from(self.graph.keys())
        for u, neighbors in self.graph.items():
            for v, edges in neighbors.items():
                for length, route in edges:
                    G.add_edge(u, v, weight=length, route=route)
        return G

    def find_candidate_stops(self, lat, lon, max_k=10, radius_m=1500):
        candidates = []
        for oid, (la, lo, _) in self.stops.items():
            d_m = math.sqrt((la - lat)**2 + (lo - lon)**2) * 111000
            if d_m > radius_m:
                continue
            candidates.append((d_m, oid))
        candidates.sort(key=lambda x: x[0])
        return [oid for _, oid in candidates[:max_k]]

    def find_path_with_metrics(self, G, start, end, max_transfers):
        pq = []
        distances = {}
        heapq.heappush(pq, (0, 0.0, 0.0, start, None))
        distances[(start, None)] = (0, 0.0, 0.0, None, None)

        best_end_state = None

        while pq:
            transfers_used, dist_so_far, walk_so_far, node, cur_route = heapq.heappop(pq)
            if transfers_used > max_transfers:
                continue
            if node == end:
                best_end_state = (node, cur_route)
                break

            rec = distances.get((node, cur_route))
            if rec is None or (rec[0], rec[1], rec[2]) < (transfers_used, dist_so_far, walk_so_far):
                continue

            for _, nbr, edge_key, attributes in G.edges(node, keys=True, data=True):
                w = attributes.get('weight', float('inf'))
                next_route = attributes.get('route', None)

                if next_route == 'walk':
                    effective_route = cur_route
                    added_transfers = 0
                    added_walk = w
                else:
                    effective_route = next_route
                    added_walk = 0
                    added_transfers = 0 if (cur_route is None or next_route == cur_route) else 1

                new_transfers = transfers_used + added_transfers
                new_dist = dist_so_far + w
                new_walk = walk_so_far + added_walk

                if new_transfers > max_transfers:
                    continue

                state = (nbr, effective_route)
                old = distances.get(state)
                alpha = 0.7
                if old is None:
                    better = True
                else:
                    old_trans, old_dist, old_walk, *_ = old
                    cmp_old = (old_trans, old_dist - alpha*old_walk, old_walk)
                    cmp_new = (new_transfers, new_dist - alpha*new_walk, new_walk)
                    better = (cmp_new < cmp_old)

                if better:
                    distances[state] = (new_transfers, new_dist, new_walk, node, cur_route)
                    heapq.heappush(pq, (new_transfers, new_dist, new_walk, nbr, effective_route))

        if best_end_state is None:
            return None, None, None, float('inf'), float('inf')

        path = []
        routes = []
        node, cur_route = best_end_state
        transfers_used, total_dist, total_walk, prev_node, prev_route = distances[(node, cur_route)]
        final_transfers = transfers_used
        final_total_dist = total_dist
        final_walk_dist = total_walk

        while True:
            path.append(node)
            rec = distances[(node, cur_route)]
            prev_node = rec[3]
            prev_route = rec[4]
            if prev_node is None:
                break
            routes.append(cur_route)
            node, cur_route = prev_node, prev_route

        path.reverse()
        routes.reverse()
        total_dist_km = final_total_dist * 111.0
        total_walk_km = final_walk_dist * 111.0

        return path, routes, final_transfers, total_dist_km, total_walk_km

    def get_route_plan(self, addrA, addrB):
        """
        Нова логіка «до 3 пересадок» із такими кроками:
        0) Дізнаємось candidate‐stops біля A і біля B (≤300 м).
        1) Перевіряємо: чи є спільний маршрут, який “заходить” і до stop ∈ candidates_A, і до stop ∈ candidates_B → 0 пересадок.
        2) Інакше пробуємо 1 пересадку: для кожного rt1 ∈ routes_near_A та rt2 ∈ routes_near_B шукаємо точки переходу (stop s1 із rt1 vs stop s2 із rt2, dist(s1,s2) ≤ 50 м).
           Обираємо найліпший варіант за: (a) мінімальна сумарна “кількість зупинок” (умовно оцінюємо δ між stops як ≃ (геодист/300 м)), 
           (b) якщо сегментів однаково → мінімізуємо сумарну пішу відстань = dA + d12 + dB.
        3) Якщо й 1 пересадка не знайдена, запускаємо “fallback” на класичний Dijkstra (до 3 пересадок).  
        """
        # ------------------------------- 0) Геокодування A та B -------------------------------
        la, lo = self.geocode(addrA)
        lb, ln = self.geocode(addrB)
        print(f"[DEBUG] Geocoded A: {addrA} → ({la:.6f}, {lo:.6f}); B: {addrB} → ({lb:.6f}, {ln:.6f})")

        # ------------------------------- 1) Знаходимо candidate stops біля A і B (≤300 м, top‐10) -------------------------------
        radius_walk_A = 400.0   # метрів
        radius_walk_B = 400.0
        candidates_A = []
        candidates_B = []
        for stop_id, (s_lat, s_lon, s_name) in self.stops.items():
            dA = math.sqrt((s_lat - la)**2 + (s_lon - lo)**2) * 111000
            dB = math.sqrt((s_lat - lb)**2 + (s_lon - ln)**2) * 111000
            if dA <= radius_walk_A:
                candidates_A.append((dA, stop_id))
            if dB <= radius_walk_B:
                candidates_B.append((dB, stop_id))

        candidates_A.sort(key=lambda x: x[0])
        candidates_B.sort(key=lambda x: x[0])
        # лишаємо лише top‐10 (за найменшим d)
        candidates_A = [sid for _, sid in candidates_A[:10]]
        candidates_B = [sid for _, sid in candidates_B[:10]]

        print("--- Кандидатні зупинки біля A (до 10) ---")
        for sid in candidates_A:
            print(f"    {sid} → «{self.stops[sid][2]}»")
        print("--- Кандидатні зупинки біля B (до 10) ---")
        for sid in candidates_B:
            print(f"    {sid} → «{self.stops[sid][2]}»")
        print("--------------------------------------")

        # Якщо жодної candidate стопи для однієї з точок, — відразу помилка
        if not candidates_A or not candidates_B:
            raise ValueError("No candidate stops near A or B (≥300m)")

        # ------------------------------- 2) Готуємо множини маршрутів біля A і біля B -------------------------------
        routes_near_A = set()
        for sA in candidates_A:
            for rt in self.routes_by_stop.get(sA, []):
                if rt != 'walk':
                    routes_near_A.add(rt)

        routes_near_B = set()
        for sB in candidates_B:
            for rt in self.routes_by_stop.get(sB, []):
                if rt != 'walk':
                    routes_near_B.add(rt)

        # ------------------------------- 3) СПРОБУЄМО 0 пересадок -------------------------------
        common_zero = routes_near_A.intersection(routes_near_B)
        if common_zero:
            best_direct = None  # (route_id, sA, dA, sB, dB)
            for rt in common_zero:
                # Знайдемо пару (sA ∈ candidates_A ∩ stops_by_route[rt],  sB ∈ candidates_B ∩ stops_by_route[rt]) 
                # із мінімальною sum(dA + dB).
                stops_rt = set(self.stops_by_route.get(rt, []))
                best_pair = None
                for sA in candidates_A:
                    if sA not in stops_rt:
                        continue
                    dA = math.sqrt((self.stops[sA][0] - la)**2 + (self.stops[sA][1] - lo)**2) * 111000
                    for sB in candidates_B:
                        if sB not in stops_rt:
                            continue
                        dB = math.sqrt((self.stops[sB][0] - lb)**2 + (self.stops[sB][1] - ln)**2) * 111000
                        total_walk = dA + dB
                        if best_pair is None or total_walk < (best_pair[1] + best_pair[3]):
                            best_pair = (sA, dA, sB, dB)
                if best_pair:
                    sA, dA, sB, dB = best_pair
                    total_walk = dA + dB
                    if best_direct is None or total_walk < (best_direct[2] + best_direct[4]):
                        best_direct = (rt, sA, dA, sB, dB)

            if best_direct:
                # Ми точно маємо 0 пересадок
                route_id, sA, dA, sB, dB = best_direct
                # Беремо shape повністю, зливаємо, обрізаємо від sA до sB:
                full_segments = fetch_route_shape(route_id, static_map.get(route_id, [{}])[0].get("bodyType"))
                merged = merge_segments(full_segments)

                # Якщо цей merged лінійний, обрізаємо:
                cropped = crop_single_segment_between_stops([merged], sA, sB, self.stops)
                if not cropped:
                    # Якщо crop не зміг (трапляються неточності), то просто з'єднуємо пряму двома точками:
                    lat1, lon1, _ = self.stops[sA]
                    lat2, lon2, _ = self.stops[sB]
                    cropped = [[(lat1, lon1), (lat2, lon2)]]

                instr = []
                instr.append(
                    f"Сісти на маршрут {route_id} біля точки A (пішки {int(dA)}м) у зупинці «{self.stops[sA][2]}», "
                    f"їхати до «{self.stops[sB][2]}» ({len(cropped[0]) - 1} зупинок), "
                    f"вихід біля точки B (пішки {int(dB)}м)."
                )

                bodyType = static_map.get(route_id, [{}])[0].get("bodyType", "bus")
                segments_info = [(route_id, bodyType, sA, sB)]
                return instr, segments_info

        # ------------------------------- 4) СПРОБУЄМО 1 пересадку -------------------------------
        best_transfer = None
        # Формат кожного кандидата: 
        # ( total_stop_count, total_walk_m, rt1, sA, s1, s2, rt2, sB, dA, d12, dB )
        positions = {sid: (lat, lon) for sid, (lat, lon, _) in self.stops.items()}

        # Поріг для “переходу” між маршрутами
        TRANSFER_RADIUS = 50.0  # метри

        for rt1 in routes_near_A:
            stops1 = self.stops_by_route.get(rt1, [])
            for rt2 in routes_near_B:
                if rt1 == rt2:
                    continue

                stops2 = self.stops_by_route.get(rt2, [])
                for s1 in stops1:
                    lat1, lon1 = positions[s1]
                    for s2 in stops2:
                        lat2, lon2 = positions[s2]
                        d12 = math.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2) * 111000
                        if d12 > TRANSFER_RADIUS:
                            continue

                        # Перше – знайдемо найкращий sA серед candidates_A ∩ stops1:
                        for sA in candidates_A:
                            if sA not in stops1:
                                continue
                            latA, lonA = positions[sA]
                            sp_dist1_m = math.sqrt((lat1 - latA)**2 + (lon1 - lonA)**2) * 111000
                            # Умовна кількість «зупинок» у сегменті rt1 між sA і s1
                            seg1_count = max(1, int(sp_dist1_m / 300))

                            # Тепер знайдемо найкращий sB серед candidates_B ∩ stops2:
                            for sB in candidates_B:
                                if sB not in stops2:
                                    continue
                                latB, lonB = positions[sB]
                                sp_dist2_m = math.sqrt((lat2 - latB)**2 + (lon2 - lonB)**2) * 111000
                                seg2_count = max(1, int(sp_dist2_m / 300))

                                # Від A до sA:
                                dA = math.sqrt((positions[sA][0] - la)**2 + (positions[sA][1] - lo)**2) * 111000
                                # Від s2 до B:
                                dB = math.sqrt((positions[sB][0] - lb)**2 + (positions[sB][1] - ln)**2) * 111000

                                total_stops = seg1_count + seg2_count
                                total_walk_m = dA + d12 + dB

                                candidate = (
                                    total_stops, total_walk_m,
                                    rt1, sA, s1, s2,
                                    rt2, sB, dA, d12, dB
                                )
                                if best_transfer is None or (candidate[0], candidate[1]) < (best_transfer[0], best_transfer[1]):
                                    best_transfer = candidate

        if best_transfer:
            (_, _,
             rt1, sA, s1, s2,
             rt2, sB, dA, d12, dB) = best_transfer

            instr = []
            # 1) Сісти на rt1:
            instr.append(
                f"Сісти на маршрут {rt1} біля точки A (пішки {int(dA)}м) у зупинці «{self.stops[sA][2]}», "
                f"їхати до «{self.stops[s1][2]}» ({max(1, int(math.ceil(d12 / 300)))} умовних зупинок)."
            )
            # 2) Пішки перейти між s1 → s2:
            instr.append(
                f"Пішки перейти від «{self.stops[s1][2]}» до «{self.stops[s2][2]}» ({int(d12)}м)."
            )
            # 3) Сісти на rt2:
            instr.append(
                f"Сісти на маршрут {rt2} у зупинці «{self.stops[s2][2]}», їхати до «{self.stops[sB][2]}» "
                f"({max(1, int(math.ceil( math.sqrt((positions[sB][0]-positions[s2][0])**2 + (positions[sB][1]-positions[s2][1])**2)*111000 / 300 )))} умовних зупинок), "
                f"вихід біля точки B (пішки {int(dB)}м)."
            )

            bt1 = static_map.get(rt1, [{}])[0].get("bodyType", "bus")
            bt2 = static_map.get(rt2, [{}])[0].get("bodyType", "bus")
            segments_info = [
                (rt1, bt1, sA, s1),
                (rt2, bt2, s2, sB)
            ]
            return instr, segments_info

        # ------------------------------- 5) Якщо 0 чи 1 пересадка не спрацювали → fallback до Dijkstra (≤3 пересадки) -------------------------------
        ###
        # Формуємо MultiDiGraph (як раніше) і запускаємо find_path_with_metrics(G, start, end, max_transfers=3).
        G = self.build_transport_graph()
        # серед candidate початкових зупинок беремо лиш ті, що існують у G.nodes()
        origin_candidates = [s for s in candidates_A if s in G]
        dest_candidates   = [s for s in candidates_B if s in G]
        if not origin_candidates or not dest_candidates:
            raise ValueError("No path found within 3 transfers (відсутні вузли у графі)")

        # Запускаємо пошук «до 3 пересадок»:
        best_overall = None
        # best_overall = (route_path, transfers, walk_km, stops_path)
        for o_stop in origin_candidates:
            for d_stop in dest_candidates:
                path, routes, transfers, dist_km, walk_km = self.find_path_with_metrics(G, o_stop, d_stop, 3)
                if path is None:
                    continue
                # Формуємо candidate: спершу мінімізуємо transfers, потім dist_km
                if best_overall is None or (transfers, dist_km) < (best_overall[1], best_overall[2]):
                    best_overall = (path, transfers, dist_km, walk_km, routes)

        if not best_overall:
            raise ValueError("No path found within 3 transfers")

        # Розпаковуємо найкращий варіант:
        path, transfers, dist_km, walk_km, routes = best_overall

        # Тепер з path і routes потрібно зібрати інструкцію
        # path = [stop0, stop1, stop2, … stopN], routes = [r1, r2, … rK]
        # transfers = K (кількість змін маршруту), де кожен r_i відповідає сегменту між stopX та stopY

        instr = []
        segments_info = []
        i = 0
        N = len(path)
        while i < N - 1:
            u = path[i]
            v = path[i+1]
            # Знаходимо атрибут 'route' у першому ребрі (u→v)
            edge_dict_uv = G[u][v]
            first_key_uv = next(iter(edge_dict_uv))
            current_route = edge_dict_uv[first_key_uv]['route']
            if current_route == 'walk':
                i += 1
                continue

            board_idx = i
            j = i
            # Рухаємося, поки next edge має той самий current_route
            while True:
                if j + 1 >= N:
                    break
                edge_dict_uv = G[path[j]][path[j+1]]
                key_uv = next(iter(edge_dict_uv))
                route_uv = edge_dict_uv[key_uv]['route']
                if route_uv == current_route:
                    j += 1
                    continue

                # Перевіримо «walk + current_route» з Clip_threshold = 20 м
                if j + 2 < N:
                    edge_uv_walk = G[path[j]][path[j+1]]
                    key_uv_w = next(iter(edge_uv_walk))
                    route_uv_w = edge_uv_walk[key_uv_w]['route']

                    edge_vv = G[path[j+1]][path[j+2]]
                    key_vv = next(iter(edge_vv))
                    route_vv = edge_vv[key_vv]['route']

                    if route_uv_w == 'walk' and route_vv == current_route:
                        la_mid, lo_mid, _ = self.stops[path[j+1]]
                        la_prev, lo_prev, _ = self.stops[path[j]]
                        dist_walk_m = math.sqrt((la_mid - la_prev)**2 + (lo_mid - lo_prev)**2) * 111000
                        if dist_walk_m <= CLIP_THRESHOLD_METERS:
                            j += 2
                            continue

                break

            alight_idx = j
            board_stop = path[board_idx]
            alight_stop = path[alight_idx]
            stops_count = alight_idx - board_idx

            # Пішки від точки A/B до відповідних board/alight вже включені:
            # просто текст:
            instr.append(
                f"Сісти на маршрут {current_route} у зупинці «{self.stops[board_stop][2]}», "
                f"їхати до «{self.stops[alight_stop][2]}» ({stops_count} зупинок)."
            )
            bodyType = static_map.get(current_route, [{}])[0].get("bodyType", "bus")
            segments_info.append((current_route, bodyType, board_stop, alight_stop))
            i = alight_idx + 1

        return instr, segments_info




    def _fetch_stops(self):
        params = {
            "where": "1=1",
            "outFields": "objectid,name,code1",
            "returnGeometry": "true",
            "f": "geojson"
        }
        r = requests.get(STOPS_LAYER, params=params, headers=GEOCODE_HEADERS)
        r.raise_for_status()
        fe = r.json().get("features", [])
        stops = {}
        for f in fe:
            p = f.get("properties", {})
            oid = str(p.get("code1") or p.get("objectid"))
            coords = f.get("geometry", {}).get("coordinates", [])
            if not coords or len(coords) < 2:
                continue
            lon, lat = coords
            name = p.get("name", "")
            stops[oid] = (lat, lon, name)
        return stops

    def _build_graph(self):
        graph = { oid: {} for oid in self.stops }
        from collections import defaultdict
        route_to_pairs = defaultdict(list)

        def seglen(path):
            total = 0.0
            for (lon1, lat1), (lon2, lat2) in zip(path, path[1:]):
                total += ((lat2 - lat1)**2 + (lon2 - lon1)**2)**0.5
            return total

        for lid in ROUTE_LAYERS:
            url = f"{FEATURESERVER_BASE}/{lid}/query"
            params = {
                "where": "1=1",
                "outFields": "from_code1,to_code1,num_route",
                "returnGeometry": True,
                "f": "geojson"
            }
            try:
                resp = requests.get(url, params=params, headers=GEOCODE_HEADERS, timeout=5)
                resp.raise_for_status()
            except Exception as e:
                print(f"[ERROR] Cannot load layer {lid}: {e}")
                continue

            data = resp.json()
            feats = data.get("features", [])
            print(f"[DEBUG] Layer {lid}: features count = {len(feats)}")
            if not feats:
                continue

            for feat in feats:
                props = feat.get("properties", {})
                a = str(props.get("from_code1") or "").strip()
                b = str(props.get("to_code1") or "").strip()
                if not a or not b or (a not in self.stops) or (b not in self.stops):
                    continue

                total_length = 0.0
                geom = feat.get("geometry", {})
                paths = geom.get("paths", None)
                if paths:
                    for path in paths:
                        total_length += seglen(path)
                else:
                    coords = geom.get("coordinates", [])
                    if coords and isinstance(coords[0], list) and isinstance(coords[0][0], list):
                        for line in coords:
                            total_length += seglen(line)
                    else:
                        total_length += seglen(coords)

                raw_route = str(props.get("num_route", "") or "").strip()
                route = raw_route.split("_")[0].strip()
                if not route:
                    continue

                route_to_pairs[route].append((a, b, total_length))
                graph[a].setdefault(b, []).append((total_length, route))
                graph[b].setdefault(a, []).append((total_length, route))

        for route, pairs in route_to_pairs.items():
            stops_set = set()
            for a, b, _ in pairs:
                stops_set.add(a)
                stops_set.add(b)

            stops_list = list(stops_set)
            for i in range(len(stops_list)):
                s1 = stops_list[i]
                la1, lo1, _ = self.stops[s1]
                for j in range(i+1, len(stops_list)):
                    s2 = stops_list[j]
                    la2, lo2, _ = self.stops[s2]
                    d_sq = (la1 - la2)**2 + (lo1 - lo2)**2
                    if d_sq <= (50 / 111000.0)**2:
                        existing_routes = [rt for (_, rt) in graph[s1].get(s2, [])]
                        if route not in existing_routes:
                            dist = math.sqrt(d_sq)
                            graph[s1].setdefault(s2, []).append((dist, route))
                            graph[s2].setdefault(s1, []).append((dist, route))

        stops_items = list(self.stops.items())
        threshold_sq = (400 / 111000.0) ** 2
        for i, (oid1, (la1, lo1, _)) in enumerate(stops_items):
            for j in range(i+1, len(stops_items)):
                oid2, (la2, lo2, _) = stops_items[j]
                d_sq = (la1 - la2)**2 + (lo1 - lo2)**2
                if d_sq <= threshold_sq:
                    dist = math.sqrt(d_sq)
                    graph[oid1].setdefault(oid2, []).append((dist, 'walk'))
                    graph[oid2].setdefault(oid1, []).append((dist, 'walk'))

        return graph

    def geocode(self, addr):
        r = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": addr, "format": "json", "limit": 1},
            headers=GEOCODE_HEADERS
        )
        r.raise_for_status()
        data = r.json()
        if not data:
            raise ValueError("Адресу не знайдено")
        return float(data[0]["lat"]), float(data[0]["lon"])

    @mainthread
    def on_route(self, addrA, addrB):
        try:
            plan, segments_info = self.get_route_plan(addrA, addrB)
        except ValueError as e:
            Popup(title="Помилка",
                  content=Label(text=str(e)),
                  size_hint=(.6, .3)).open()
            return

        # Забираємо старі шари / маркери, якщо вони є
        if hasattr(self, 'route_layers'):
            for layer in self.route_layers:
                # 1) Видаляємо шар із MapView
                self.mapview.remove_layer(layer)
                # 2) Одразу ж відписуємо метод reposition від подій карти
                self.mapview.unbind(on_zoom=layer.reposition)
                self.mapview.unbind(on_map_relocated=layer.reposition)
        self.route_layers = []

        if hasattr(self, 'stop_markers'):
            for sm in self.stop_markers:
                self.mapview.remove_marker(sm)
        self.stop_markers = []

        # Тепер по кожному сегменту складаємо: green-marker, yellow-marker, route-line
        for idx_info, (route, bodyType, board_stop, alight_stop) in enumerate(segments_info):
            full_segments = fetch_route_shape(route, bodyType)
            merged_poly = merge_segments(full_segments)
            cropped = crop_single_segment_between_stops(
                [merged_poly], board_stop, alight_stop, self.stops
            )
            if not cropped:
                print(f"[WARNING] Cropping failed for route {route} ({board_stop}->{alight_stop}), drawing straight line")
                la1, lo1, _ = self.stops[board_stop]
                la2, lo2, _ = self.stops[alight_stop]
                cropped = [[(la1, lo1), (la2, lo2)]]

            # Колір за типом транспорту
            color = {
                "тролейбус": TROLLEY_COLOR,
                "трамвай":   TRAM_COLOR,
                "маршрутка": MARSHRUTKA_COLOR
            }.get(bodyType, BUS_COLOR)

            # Зелена мітка (посадка) – з текстом route
            la_b, lo_b, _ = self.stops[board_stop]
            green_marker = StopMarker(lat=la_b, lon=lo_b, text=route, color=(0, 1, 0, 1))
            self.mapview.add_marker(green_marker)
            self.stop_markers.append(green_marker)

            # Жовта мітка (висадка) – без тексту, просто кружок
            la_a, lo_a, _ = self.stops[alight_stop]
            yellow_marker = StopMarker(lat=la_a, lon=lo_a, text="", color=(1, 1, 0, 1))
            self.mapview.add_marker(yellow_marker)
            self.stop_markers.append(yellow_marker)

            # Малюємо самим RouteLayer
            layer = RouteLayer(cropped, color)
            self.mapview.add_layer(layer)
            # Перемальовуємо, коли змінилась zoom/координати
            self.mapview.bind(
                on_zoom=layer.reposition,
                on_map_relocated=layer.reposition
            )
            layer.reposition()
            self.route_layers.append(layer)

        # Відобразимо текстовий опис маршруту
        content = BoxLayout(orientation='vertical', spacing=4, padding=8)
        for step in plan:
            content.add_widget(Label(text=step, size_hint_y=None, height='24dp'))
        Popup(title="Маршрут", content=content, size_hint=(.8, .6)).open()


    @mainthread
    def show_route(self, route_id, bodyType):
        if hasattr(self, 'route_layers'):
            for layer in self.route_layers:
                self.mapview.remove_layer(layer)
            self.route_layers = []

        segments = fetch_route_shape(route_id, bodyType)
        if not segments:
            print(f"[WARNING] Empty shape for route {route_id}")
            return

        color = {
            "тролейбус": TROLLEY_COLOR,
            "трамвай":   TRAM_COLOR,
            "маршрутка": MARSHRUTKA_COLOR
        }.get(bodyType, BUS_COLOR)

        layer = RouteLayer(segments, color)
        self.mapview.add_layer(layer)
        layer.reposition()
        self.route_layers = [layer]


class TrackApp(App):
    def build(self):
        Builder.load_string(KV)
        return RootWidget()


if __name__ == "__main__":
    TrackApp().run()
