
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from itertools import product

# Charger les données réelles
data = pd.read_csv('populations_lapins_renards.csv', parse_dates=['date'])

# Afficher les données
def evaluate_model(params, data):
    alpha, beta, delta, gamma = params
    time = [0]
    lapin = [data['prey_population'][0]]
    renard = [data['predator_population'][0]]


    step = 0.001

    for i in range(1, len(data)):
        new_values_time = time[-1] + step
        new_values_lapin = (lapin[-1] + alpha * lapin[-1] - beta * lapin[-1] * renard[-1]) * step + lapin[-1]
        new_values_renard = (delta * lapin[-1] * renard[-1] - gamma * renard[-1]) * step + renard[-1]

        time.append(new_values_time)
        lapin.append(new_values_lapin)
        renard.append(new_values_renard)

    lapin = np.array(lapin)
    lapin *= 1000
    renard = np.array(renard)
    renard *= 1000

   

