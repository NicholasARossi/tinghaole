# import required libraries
# import sounddevice as sd
# from scipy.io.wavfile import write
# import wavio as wv

import pandas as pd
import numpy as np

ma_test_set=pd.read_csv("../../data/ma_data/ma_encodings.csv")



#randomized index
index=np.random.choice(np.arange(len(ma_test_set)))


ma_test_set.iloc[index]




# # Sampling frequency
# freq = 44100
#
# # Recording duration
# duration = 5
#
# # Start recorder with the given values
# # of duration and sample frequency
# recording = sd.rec(int(duration * freq),
#                    samplerate=freq, channels=2)
#
# # Record audio for the given number of seconds
# sd.wait()
#
# # This will convert the NumPy array to an audio
# # file with the given sampling frequency
# write("recording0.wav", freq, recording)
#
# # Convert the NumPy array to audio file
# wv.write("recording1.wav", recording, freq, sampwidth=2)

