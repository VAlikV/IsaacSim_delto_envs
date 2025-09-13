import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QSlider, QLabel
from PyQt5.QtCore import Qt
import socket


class SliderDemo(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("5 Sliders Demo")
        self.layout = QVBoxLayout()

        self.sliders = []
        self.labels = []

        for i in range(20):
            label = QLabel(f"Slider {i+1}: 0")
            slider = QSlider(Qt.Horizontal)
            slider.setRange(-100, 100)
            slider.setValue(0)

            # Привязываем функцию при изменении значения
            slider.valueChanged.connect(self.on_value_change)

            self.layout.addWidget(label)
            self.layout.addWidget(slider)

            self.sliders.append(slider)
            self.labels.append(label)

        self.setLayout(self.layout)

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("127.0.0.1", 8080))
        self.sock.settimeout(1.0)


    def on_value_change(self):
        """Функция вызывается при изменении значения слайдера"""

        pos = []

        for i, l in enumerate(self.labels):
            l.setText(f"Slider {i+1}: {self.sliders[i].value()}")

            pos.append(self.sliders[i].value()/100)
        
        print(pos)
        self.sock.sendto(str(pos).encode('utf-8'), ("127.0.0.1", 8081))



if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SliderDemo()
    window.show()
    sys.exit(app.exec_())