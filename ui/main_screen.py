# ui/main_screen.py
import math
from kivy.clock             import Clock
from kivy.uix.boxlayout     import BoxLayout
from kivy.uix.screenmanager import Screen
from kivy.graphics          import Color, Ellipse
from kivy.uix.label         import Label

from kivy_garden.mapview    import MapView, MarkerMapLayer, MapMarker
from logic.transport_data   import load_data


def _geo_distance_m(p1, p2):
    lat1, lon1 = p1['lat'], p1['lon']
    lat2, lon2 = p2['lat'], p2['lon']
    dy = (lat2 - lat1) * 111000
    avg_lat = math.radians((lat1 + lat2) / 2)
    dx = (lon2 - lon1) * 111000 * math.cos(avg_lat)
    return math.hypot(dx, dy)


class NumberedMarker(MapMarker):
    def __init__(self, number, color_rgba, **kwargs):
        super().__init__(source='', **kwargs)
        self.number     = number
        self.small_size = (10, 10)
        self.big_base   = (30, 30)
        self.threshold  = 14

        self.size     = self.small_size
        self.anchor_x = self.size[0] / 2
        self.anchor_y = self.size[1] / 2

        with self.canvas:
            Color(*color_rgba)
            self._ellipse = Ellipse(pos=self.pos, size=self.size)

        self._label = Label(
            text=str(self.number),
            font_size=0,
            size_hint=(None, None),
            opacity=0
        )
        self.add_widget(self._label)
        self.bind(pos=self._update, size=self._update)

    def _update(self, *args):
        self._ellipse.pos  = self.pos
        self._ellipse.size = self.size
        self._label.center = self.center

    def update_visual(self, zoom):
        if zoom >= self.threshold:
            f = zoom / 12.0
            w, h = self.big_base[0]*f, self.big_base[1]*f
            self.size           = (w, h)
            self._label.font_size = 12 * f
            self._label.opacity   = 1
        else:
            self.size            = self.small_size
            self._label.opacity  = 0

        self.anchor_x = self.size[0] / 2
        self.anchor_y = self.size[1] / 2


class TransportLayer(MarkerMapLayer):
    def __init__(self, speed_m_s=10, **kwargs):
        super().__init__(**kwargs)
        self.speed  = speed_m_s
        self._infos = []

    def register(self, marker, route):
        self._infos.append({'marker': marker, 'route': route, 'idx': 0})

    def start(self, interval=1):
        Clock.schedule_interval(self._move_all, interval)

    def _move_all(self, dt):
        for info in self._infos:
            info['idx'] = (info['idx'] + 1) % len(info['route'])
            pt = info['route'][info['idx']]
            info['marker'].lat = pt['lat']
            info['marker'].lon = pt['lon']
        self.reposition()

    def reposition(self):
        zoom = self.parent.zoom
        for info in self._infos:
            info['marker'].update_visual(zoom)
        super().reposition()


class MainScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # 1) ПОВНІСТЮ РОЗПИСУЄМО UI
        root = BoxLayout(orientation='vertical')

        # Саме тут визначаємо self.map_view
        self.map_view = MapView(zoom=12, lat=50.4501, lon=30.5234)
        root.add_widget(self.map_view)

        # Додатковий контрол (тут просто місце для пізнішого поля/кнопок)
        # ctrl = BoxLayout(size_hint=(1, 0.1))
        # root.add_widget(ctrl)

        self.add_widget(root)

        # 2) ЗАВАНТАЖУЄМО ДАНІ І СТВОРЮЄМО ШАР
        transports = load_data()
        layer = TransportLayer(speed_m_s=10)
        self.map_view.add_layer(layer)

        # 3) ДОДАЄМО МАРКЕРИ ЧЕРЕЗ self.map_view.add_marker
        for tr in transports:
            color = (0,0,1,1) if tr['type']=='marshrutka' else (1,0.5,0,1)
            m = NumberedMarker(
                number=tr['number'],
                color_rgba=color,
                lat=tr['coords'][0]['lat'],
                lon=tr['coords'][0]['lon']
            )
            # додаємо в карту, вказуємо шар
            self.map_view.add_marker(m, layer=layer)
            layer.register(m, tr['coords'])

        # 4) Початкове позиціювання під стартовий зум
        layer.reposition()
        # 5) І запускаємо рух одразу
        layer.start()

