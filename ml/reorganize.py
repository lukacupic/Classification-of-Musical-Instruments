import os
from tqdm import tqdm
import pandas as pd
import numpy as np
from matplotlib import cm
from scipy.io import wavfile

def main():
    df = pd.read_csv("instruments.csv")

    for i, row in df.iterrows():
        fn = row.fname
        c = row.label

        path = "wavfiles2/" + c

        if not os.path.exists(path):
            os.mkdir(path)

        rate, signal = wavfile.read("wavfiles/" + fn)
        wavfile.write(filename=path + "/" + fn + fn, rate=rate, data=signal)

if __name__ == '__main__':
    main()
