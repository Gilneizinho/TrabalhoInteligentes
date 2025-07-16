from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel
import torchaudio
import random
from gtts import gTTS
import os, atexit
import torch
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time
import joblib

i = 0
pasta = os.path.dirname(os.path.abspath(__file__))

model_path = "meu_modelo_whisper"
modelo_base = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
processor = WhisperProcessor.from_pretrained(model_path)
model = PeftModel.from_pretrained(modelo_base, model_path)
model.eval()
classifier = joblib.load('classificador.pkl')

respostas = {
    "ligar_luz": [
        "Entendido, ligando a luz.",
        "Ok, luz ligada.",
        "Luz do seu quarto acionada.",
        "Ligando a luz agora."
    ],
    "desligar_luz": [
        "Entendido, desligando a luz.",
        "Ok, luz desligada.",
        "Apagando a luz para você.",
        "Luz do quarto desligada."
    ],
    "ligar_ventilador": [
        "Ligando o ventilador.",
        "Ventilador ligado.",
        "Ventilador ligado com sucesso."
    ],
    "desligar_ventilador": [
        "Desligando o ventilador.",
        "Ventilador desligado.",
        "Ventilador desligado com sucesso."
    ],
    "tocar_musica": [
        "OK, colocando para tocar uma música.",
        "Tocando a música agora.",
    ],
    "parar_musica": [
        "OK, parando a música atual.",
        "OK, música parada.",
        "Música desligada.",
    ],
}

def falar_resposta(intencao):
    global i
    if intencao in respostas:
        frase = random.choice(respostas[intencao])
    else:
        frase = "Desculpe, não entendi sua solicitação."
    tts = gTTS(text=frase, lang='pt-br')
    tts.save(f"resposta{i}.mp3")
    os.system(f"start resposta{i}.mp3")
    i+=1

def transcrever(audio_path):
    print(f"Transcrevendo: {audio_path}")
    waveform, sample_rate = torchaudio.load(audio_path)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
    inputs = processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")
    input_features = inputs["input_features"]
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="portuguese", task="transcribe")
    with torch.no_grad():
        predicted_ids = model.generate(
            input_features,
            forced_decoder_ids=forced_decoder_ids
        )
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    print("Transcrição:", transcription)
    intencao_detectada = classifier.predict([transcription])[0]
    print(intencao_detectada)
    falar_resposta(intencao_detectada)

class AudioHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.src_path.endswith(".wav"):
            time.sleep(0.5)
            comando = transcrever(event.src_path)
            print(f'comando = {comando}')

def cleanup():
    print("Limpando arquivos temporários...")
    for arquivo in os.listdir(pasta):
        if arquivo.startswith("resposta") and arquivo.endswith(".mp3"):
            caminho = os.path.join(pasta, arquivo)
            try:
                os.remove(caminho)
                print(f"Deletado: {arquivo}")
            except Exception as e:
                print(f"Erro ao deletar {arquivo}: {e}")

path = r"G:\PycharmProjects\tetsst"

event_handler = AudioHandler()
observer = Observer()
observer.schedule(event_handler, path=path, recursive=False)
observer.start()

print(f"IA carregada e monitorando a pasta: {path}")

atexit.register(cleanup)
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    observer.stop()
observer.join()