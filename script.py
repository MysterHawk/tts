from bark import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav
from IPython.display import Audio
import os

os.environ["SUNO_OFFLOAD_CPU"] = "True"
os.environ["SUNO_USE_SMALL_MODELS"] = "True"

# download and load all models
preload_models()

# generate audio from text
text_prompt = """
     Hello, my name is Suno. And, uh — I like pizza. [laughs] 
     But I also have other interests such as playing tic tac toe & your mom.
"""

# audio_array = generate_audio(text_prompt, history_prompt="v2/en_speaker_6")
audio_array = generate_audio(text_prompt, history_prompt="v2/it_speaker_1")

# save audio to disk
write_wav("bark_generation.wav", SAMPLE_RATE, audio_array)
  
# play text in notebook
Audio(audio_array, rate=SAMPLE_RATE)