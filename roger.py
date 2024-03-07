# %%

from uuid import UUID
from langchain_core.outputs import LLMResult
import torch

from transformers import pipeline
import time

from llama_cpp import Llama
#from huggingface_hub import hf_hub_download
import numpy as np


import requests
import sounddevice as sd


import wave
#import pyaudio




import threading
import time
import queue
import sys
import re

from queue import Queue
from flask_socketio import SocketIO, emit
#from jetsonGPTmain import webserver

import ollama


from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import ConversationChain
from langchain.callbacks.base import BaseCallbackHandler
from langchain_community.llms import LlamaCpp
# %%
#pip install torch
#pip install transformers
#!pip install sounddevice
#!pip install pydub
#!pip install espeak_phonemizer

# %%
def print_with_newline(string):
    words = string.split()
    for i in range(0, len(words), 10):
        print(' '.join(words[i:i+10]))

def replace_apostrophes(string):
    return string.replace('"', "'")

# %%
#!pip install llama-cpp-python
#!pip install piper-tts
#!pip install pyaudio

# %%
#model = hf_hub_download(repo_id="TheBloke/phi-2-dpo-GGUF", filename="phi-2-dpo.Q5_K_S.gguf")

# %%



#pipe = pipeline("automatic-speech-recognition", "openai/whisper-tiny")
#pipe1 =  pipeline("automatic-speech-recognition", "openai/whisper-base")
#pipe2 = pipeline("automatic-speech-recognition", "openai/whisper-small")
#pipe3 = pipeline("automatic-speech-recognition", "openai/whisper-medium")
#pipe4 = pipeline("automatic-speech-recognition", "openai/whisper-large-v2")
# {'text': "GOING ALONG SLUSHY COUNTRY ROADS AND SPEAKING TO DAMP AUDIENCES IN DRAUGHTY SCHOOL ROOMS DAY AFTER DAY FOR A FORTNIGHT HE'LL HAVE TO PUT IN AN APPEARANCE AT SOME PLACE OF WORSHIP ON SUNDAY MORNING AND HE CAN COME TO US IMMEDIATELY AFTERWARDS"}


# %%





        
def record_audio(file_path, duration=5, sample_rate=44100):
    print(f"Recording audio for {duration} seconds...")

    # Initialize PyAudio
    audio = pyaudio.PyAudio()

    # Define audio stream parameters
    chunk_size = 1024  # Adjust the chunk size as needed
    format = pyaudio.paInt16
    channels = 2

    # Open audio stream
    stream = audio.open(format=format,
                        channels=channels,
                        rate=sample_rate,
                        input=True,
                        frames_per_buffer=chunk_size)

    frames = []
    # Record audio for the specified duration
    for _ in range(0, int(sample_rate / chunk_size * duration)):
        data = stream.read(chunk_size)
        
        
        frames.append(data)

    
    # Stop and close the audio stream
    stream.stop_stream()
    stream.close()
    audio.terminate()

    print("Recording complete. Saving to file.")

    array_data = np.frombuffer(b''.join(frames), dtype=np.float16)
    #b  = pipe1({"sampling_rate":  44100, "raw": array_data})['text']
    #print(b)

    # Save audio data to a WAV file
    with wave.open(file_path, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(audio.get_sample_size(format))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))

    print(f"Audio saved to {file_path}")




# %%
#!pip install torchaudio

# %%


# %%
#To get the event from the frontEnt to stop generation
global_stop_event = threading.Event()

def my_function(thread_lock, thread_id, sentence):
    with thread_lock:

        if not global_stop_event.is_set():

            st = time.time()
            # Read audio data from stdin
            data = webserver.predict(sentence)
            end = time.time()
            #print("script", end - st)
            npa = np.asarray(data['data'], dtype=np.int16)
            if not global_stop_event.is_set():
                sd.play(npa, data['sample-rate'], blocking=True)


def worker(thread_lock, thread_id, sentence_queue):
    while True:
        sentence = sentence_queue.get()
        if sentence is None:
            break
        my_function(thread_lock, thread_id, sentence)
        sentence_queue.task_done()




class MyCustomHandler(BaseCallbackHandler):
    

    output = """ """
    
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        emit('sentence', {'message': token})
        self.output += token
        


    def on_llm_end(self, response: LLMResult, *, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: LlamaCpp) -> LlamaCpp:
        
        return super().on_llm_end(response, run_id=run_id, parent_run_id=parent_run_id, **kwargs)






model = "./phi-2-dpo.Q5_K_S.gguf"
llm = LlamaCpp(
    model_path=model,
    temperature=0.75,
    max_tokens=50,
    top_p=1,
    streaming=True,
    callbacks=[MyCustomHandler()],
    verbose=True,  # Verbose is required to pass to the callback manager
)

from langchain_community.llms import Ollama
ollama = Ollama(model="mistral", callbacks=[MyCustomHandler()])


conversation_with_summary = ConversationChain(
    llm=llm,
    # We set a very low max_token_limit for the purposes of testing.
    memory = ConversationSummaryBufferMemory(llm=ollama, max_token_limit=500),
    verbose = True,
    
    
)



def modelRunWithMemory(textinput=False, input=""):

    if textinput == True:
        QUERY = input
    else:
        record_audio("recorded_audio.wav", duration=6)
        QUERY = pipe1("recorded_audio.wav")['text']
    print(QUERY)

    conversation_with_summary.invoke(QUERY)



def modelRun(textinput=False, input=""):
    
    
    Ollama = True
    #record_audio("recorded_audio.wav", duration=6)
    print("recorded")
    
    if textinput == True:
        QUERY = input
    else:
        record_audio("recorded_audio.wav", duration=6)
        QUERY = pipe1("recorded_audio.wav")['text']
    print(QUERY)


    

    output = """ """
    cur_sentence = ""

    if Ollama == True:
        stream = ollama.chat(
        model='phi',
        messages=[{'role': 'user', 'content': QUERY}],
        stream=True,
    )

        for chunk in stream:

            if global_stop_event.is_set():
                return
        
            character = chunk['message']['content']
            
            emit('sentence', character)
            output += character
            
    else:

        for token in  (llm.create_chat_completion(
            messages = [
                # {"role": "system", "content": "You are a story writing assistant."},
                {
                    "role": "user",
                    "content": QUERY + "."
                }
            ],
            stream = True,
            max_tokens = 200
        )):
            character = token['choices'][0]['delta'].get('content', "")
            emit('sentence', {'data': character})
            output += character
            

        



    print("------------------------------------------")
    print_with_newline(output)
    print("------------------------------------------")

    













