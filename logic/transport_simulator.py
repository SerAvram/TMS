from kivy.uix.label import Label
from kivy.uix.floatlayout import FloatLayout
import random

class TransportSimulator:
    def __init__(self):
        self.positions = [(0.1, 0.1), (0.2, 0.3), (0.3, 0.5), (0.4, 0.7), (0.5, 0.9)]
        self.index = 0

    def update_position(self, dt):
        if self.index >= len(self.positions):
            self.index = 0
        pos = self.positions[self.index]
        print(f"🚌 Транспорт на позиції: {pos}")
        self.index += 1
