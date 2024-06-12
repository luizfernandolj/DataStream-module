import pandas as pd
import numpy as np
from slidingWindow import SlidingWindow

window_size = 1000

train = pd.read_csv("datasets/train/AedesSex.train.csv")
stream = pd.read_csv("datasets/test/AedesSex.test.csv")

start_window = train.iloc[-window_size:]

stream = SlidingWindow(start_window=start_window, stream=stream, has_context=True)

def f(start_window, current_window):
    return start_window.labels() - current_window.labels()

for window in stream:
    print(stream(f))
