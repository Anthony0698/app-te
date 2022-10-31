from time import localtime
from pygame import mixer

H = input("Ingrese la hora")
M = input("Ingrese el minuto")

while True:
    if localtime().tm_hour == int(H) and localtime().tm_min == int(M):
        print("Sonido de Alarma")
        mixer.init()
        mixer.music.load("app-drowsiness/car-alarm.mp3")
        mixer.music.play()
        break