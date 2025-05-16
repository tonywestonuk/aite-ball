import json
import argparse
import time
import random
import serial

import subprocess
import re

import sounddevice as sd
import scipy.io.wavfile as wav
import threading
import queue
import uuid
import numpy as np
from pywhispercpp.model import Model as WhisperModel
from unidecode import unidecode

from llama_cpp import Llama
from pathlib import Path


#load models
llm_model_path = str(Path("~/aite-ball/models/gemma-3-1b-it-q4_k_m.gguf").expanduser())
whisper_model_path = str(Path("~/aite-ball/models/ggml-tiny.bin").expanduser())


llm = Llama(model_path=llm_model_path, n_threads=4)
whisper_model = WhisperModel(whisper_model_path)


SAMPLE_RATE = 16000
CHANNELS = 1
audio_queue = queue.Queue()
is_recording = False
recording_thread = None
filename = None

ser = None

whisper_model = WhisperModel("tiny.en")

def process_audio_file(nd_arr):
    ser.write('tnk1\r'.encode())
    print(f"Processing audio file: ")
    segments = whisper_model.transcribe(nd_arr)

    transc = ""
    for s in segments:
        transc +=s.text
    main(quest=transc)
    

    # Your code to process the audio file goes here.

def record_audio(q, stop_event, file_path):
    print("Recording started...")
    audio_data = []

    def callback(indata, frames, time_info, status):
        if status:
            print(f"Status: {status}", flush=True)
        audio_data.append(indata.copy())

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, callback=callback,dtype=np.int16):
        while not stop_event.is_set():
            sd.sleep(100)

    # Stack all chunks into a single NumPy array
    full_audio = np.concatenate(audio_data, axis=0)
    wav.write(file_path, SAMPLE_RATE, full_audio)
    print("Recording saved.")
    process_audio_file(file_path)


def main(quest='Should I buy the red or blue shoes'):
    ser.write('tnk2\r'.encode())
    options = ["friendly negative", "positive", "funny", "cautious", "alternative"]
    random_choice = random.choice(options)
    quest = unidecode(quest)
    print(quest)
    print(random_choice)

    output = llm(f"A {random_choice} response, less than 7 words, "
                 f", to the question: '{quest}', is \"",
    max_tokens=15,
    stop=["Q:"],
    temperature = 1,
    echo=False,
    seed=int(time.time())
   )
    
    
    print(json.dumps(output, indent=2))
    chosen=output['choices'][0]["text"]
    chosen = chosen[:chosen.find('"')]
    chosen = unidecode(chosen)
    llm.reset()

    ser.write((chosen + '\r').encode())

ANSI_ESCAPE_RE = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

def strip_ansi_codes(text):
    return ANSI_ESCAPE_RE.sub('', text)


def handle_command(text):
    text = strip_ansi_codes(text)
    print(f"Handling transcription: {text}")
    if text[0] != '(':
        main(text)


def listen_serial():
    global is_recording, recording_thread, filename

    stop_event = threading.Event()
    
    ser.write(('Ask me something!\r').encode())
    while True:
        if ser.in_waiting > 0:
            byte = ser.read().decode('utf-8')

            if byte == '0' and not is_recording:
                ser.write('lis\r'.encode())
                is_recording = True
                stop_event.clear()
                filename = "/tmp/8ball_recording.wav"
                recording_thread = threading.Thread(target=record_audio, args=(audio_queue, stop_event, filename))
                recording_thread.start()

            elif byte == '1' and is_recording:
                stop_event.set()
                recording_thread.join()
                is_recording = False

        else:
            time.sleep(.1)



if __name__ == "__main__":
    ser = serial.Serial('/dev/ttyS5', baudrate=9600, timeout=1)
    listen_serial()
    ser.close();


