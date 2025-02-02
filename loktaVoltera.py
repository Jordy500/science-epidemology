import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Charger les données réelles
data = pd.read_csv('populations_lapins_renards.csv', parse_dates=['date'])
data['jours'] = (data['date'] - data['date'].min()).dt.days

def lotka_volterra(state, t, alpha, beta, delta, gamma):
    x, y = state
    dxdt = alpha * x - beta * x * y
    dydt = delta * x * y - gamma * y
    
    return [dxdt, dydt]

def evaluate_model(params, data):
    alpha, beta, delta, gamma = params
    time = np.arange(len(data))
    prey_init = data['lapin'].iloc[0]
    pred_init = data['renard'].iloc[0]
    
    sol = odeint(lotka_volterra, [prey_init, pred_init], time, args=(alpha, beta, delta, gamma))
    predicted_prey = sol[:, 0]
    predicted_pred = sol[:, 1]
    
    mse_prey = np.mean((data['lapin'] - predicted_prey)**2)
    mse_pred = np.mean((data['renard'] - predicted_pred)**2)
    
    total_mse = mse_prey + mse_pred
    
    print(f"MSE pour les lapins : {mse_prey:.4f}")
    print(f"MSE pour les renards : {mse_pred:.4f}")
    print(f"MSE total : {total_mse:.4f}")
    
    return total_mse

# Définir les paramètres initiaux
alpha = 0.5
beta = 0.02
delta = 0.01
gamma = 0.3

# Évaluation initiale
initial_mse = evaluate_model((alpha, beta, delta, gamma), data)

# Tracer les résultats initiaux
time = np.arange(len(data))
initial_sol = odeint(lotka_volterra, [data['lapin'].iloc[0], data['renard'].iloc[0]], time, args=(alpha, beta, delta, gamma))
predicted_prey_initial = initial_sol[:, 0]
predicted_pred_initial = initial_sol[:, 1]

plt.figure(figsize=(12, 6))
plt.plot(data['jours'], data['lapin'], label='Lapins réels', color='blue')
plt.plot(data['jours'], data['renard'], label='Renards réels', color='red')
plt.plot(data['jours'], predicted_prey_initial, label='Lapins prédits initiaux', linestyle='--', color='lightblue')
plt.plot(data['jours'], predicted_pred_initial, label='Renards prédits initiaux', linestyle='--', color='pink')
plt.xlabel('jours')
plt.ylabel('Population')
plt.title('Modèle Lotka-Volterra initial')
plt.legend()
plt.grid(True)
plt.show()

# Analyse supplémentaire
print(f"\nMoyenne de la population de lapins : {data['lapin'].mean():.2f}")
print(f"Moyenne de la population de renards : {data['renard'].mean():.2f}")
print(f"Corrélation entre les populations : {data['lapin'].corr(data['renard']):.4f}")

# Identification des périodes de stabilité/fluctuation
stable_threshold = 1.5
stable_periods = data[(data['lapin'] > data['renard']) & (data['lapin'] < stable_threshold * data['renard'])]
fluctuating_periods = data[~((data['lapin'] > data['renard']) & (data['lapin'] < stable_threshold * data['renard']))]

print(f"\nPériodes stables : {len(stable_periods)} jours")
print(f"Périodes fluctuantes : {len(fluctuating_periods)} jours")

# Ajustement des paramètres
best_params = (alpha, beta, delta, gamma)
best_mse = initial_mse

# Grille de recherche pour les paramètres
param_grid = {
    'alpha': [1/3, 2/3, 1, 4/3],
    'beta': [1/3, 2/3, 1, 4/3],
    'delta': [1/3, 2/3, 1, 4/3],
    'gamma': [1/3, 2/3, 1, 4/3]
}

print("\nDébut de l'ajustement des paramètres")
for alpha in param_grid['alpha']:
    for beta in param_grid['beta']:
        for delta in param_grid['delta']:
            for gamma in param_grid['gamma']:
                params = (alpha, beta, delta, gamma)
                mse = evaluate_model(params, data)
                print(f"\nPour alpha={alpha}, beta={beta}, delta={delta}, gamma={gamma}:")
                if mse < best_mse:
                    best_mse = mse
                    best_params = params

print(f"\nFin de l'ajustement des paramètres")
print(f"Meilleurs paramètres trouvés : alpha={best_params[0]}, beta={best_params[1]}, delta={best_params[2]}, gamma={best_params[3]}")
print(f"Plus petite MSE : {best_mse:.4f}")

# Tracer les résultats avec les meilleurs paramètres
final_sol = odeint(lotka_volterra, [data['lapin'].iloc[0], data['renard'].iloc[0]], time, args=best_params)
predicted_prey_final = final_sol[:, 0]
predicted_pred_final = final_sol[:, 1]

plt.figure(figsize=(12, 6))
plt.plot(data['jours'], data['lapin'], label='Lapins réels', color='blue')
plt.plot(data['jours'], data['renard'], label='Renards réels', color='red')
plt.plot(data['jours'], predicted_prey_final, label='Lapins prédits ajustés', linestyle='--', color='lightgreen')
plt.plot(data['jours'], predicted_pred_final, label='Renards prédits ajustés', linestyle='--', color='orange')
plt.xlabel('jours')
plt.ylabel('Population')
plt.title('Modèle Lotka-Volterra ajusté')
plt.legend()
plt.grid(True)
plt.show()
