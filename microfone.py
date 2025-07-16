import sounddevice as sd
import numpy as np
import keyboard
from scipy.io.wavfile import write
import os, atexit

i = 0
pasta = os.path.dirname(os.path.abspath(__file__))

def microfone(i):
    fs = 44100
    canal = 1
    audio_buffer = []
    print("Gravando... (aperte espaço para parar)")
    def callback(indata, frames, time, status):
        audio_buffer.append(indata.copy())
    with sd.InputStream(samplerate=fs, channels=canal, callback=callback):
        while True:
            if keyboard.is_pressed('space'):
                break
    audio_array = np.concatenate(audio_buffer, axis=0)
    nome_arquivo = f'gravacao{i}.wav'
    write(nome_arquivo, fs, audio_array)
    return i + 1

def cleanup():
    print("Limpando arquivos de gravação")
    for arquivo in os.listdir(pasta):
        if arquivo.startswith("gravacao") and arquivo.endswith(".wav"):
            caminho = os.path.join(pasta, arquivo)
            try:
                os.remove(caminho)
                print(f"Deletado: {arquivo}")
            except Exception as e:
                print(f"Erro ao deletar {arquivo} : {e}")

try:
    while True:
        i = microfone(i)
        continuar = input("Deseja gravar outro áudio? (S/N) ").strip().upper()
        if continuar != "S":
            break
except KeyboardInterrupt:
    atexit.register(cleanup)