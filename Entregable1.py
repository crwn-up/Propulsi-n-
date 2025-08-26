import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ----- PARÁMETROS DEL COHETE -----
m_seca = 2.2         # kg
m_fuel_total = 0.625 # kg
Cd = 0.75
diam = 0.086         # m
rho_air = 1          # kg/m^3
g = 9.78             # m/s^2
v_e = 960            # m/s

# Área transversal
A = np.pi * (diam/2)**2

# ----- FLUJO MÁSICO -----
def mdot_fuel(t):
    return 5/16 if t <= 2 else 0

# ----- ECUACIÓN DEL MOVIMIENTO -----
def cohete(t, y):
    h, v = y
    m = m_seca + max(0, m_fuel_total - (t * mdot_fuel(t)))
    F_thrust = mdot_fuel(t) * v_e
    F_drag = 0.5 * rho_air * Cd * A * v**2
    a = (F_thrust - F_drag)/m - g
    return [v, a]

# ----- EVENTO: COHETE TOCA EL SUELO -----
def suelo(t, y):
    return y[0]
suelo.terminal = True
suelo.direction = -1

# ----- CONDICIONES INICIALES -----
y0 = [0, 0]                 
t_span = (0, 1000)           
t_eval = np.linspace(0, 1000, 3000)  

# ----- RESOLVER -----
sol = solve_ivp(cohete, t_span, y0, t_eval=t_eval, events=suelo)

# ----- CALCULOS EXTRAS -----
h = sol.y[0]
v = sol.y[1]
t = sol.t

m = m_seca + np.maximum(0, m_fuel_total - np.array([ti * mdot_fuel(ti) for ti in t]))
F_thrust = np.array([mdot_fuel(ti) * v_e for ti in t])
F_drag = 0.5 * rho_air * Cd * A * v**2
F_weight = m * g
a = (F_thrust - F_drag)/m - g

# ---- REPORTES ----
idx_meco = np.argmin(np.abs(t - 2))
h_meco, v_meco = h[idx_meco], v[idx_meco]

idx_apogeo = np.argmax(h)
t_apogeo, h_apogeo = t[idx_apogeo], h[idx_apogeo]

h_max, v_max, a_max = np.max(h), np.max(v), np.max(a)

print("===== REPORTES DEL VUELO =====")
print(f"Altura en M.E.C.O: {h_meco:.2f} m")
print(f"Velocidad en M.E.C.O: {v_meco:.2f} m/s")
print(f"Tiempo desde despegue al apogeo: {t_apogeo:.2f} s")
print(f"Altura máxima alcanzada: {h_max:.2f} m")
print(f"Velocidad máxima alcanzada: {v_max:.2f} m/s")
print(f"Aceleración máxima alcanzada: {a_max:.2f} m/s²")

# ----- GRAFICAS -----
plt.figure(figsize=(14,10))

# Altura
plt.subplot(2,2,1)
plt.plot(t, h, label="Altura (m)")
plt.scatter(t[idx_meco], h_meco, color="red", label="M.E.C.O")
plt.scatter(t_apogeo, h_apogeo, color="purple", label="Apogeo")
plt.xlabel("Tiempo (s)")
plt.ylabel("Altura (m)")
plt.title("Altura vs Tiempo")
plt.grid()
plt.legend()

# Velocidad
plt.subplot(2,2,2)
plt.plot(t, v, label="Velocidad (m/s)", color="orange")
plt.scatter(t[idx_meco], v_meco, color="red", label="M.E.C.O")
plt.scatter(t[np.argmax(v)], v_max, color="green", label="Velocidad máx")
plt.xlabel("Tiempo (s)")
plt.ylabel("Velocidad (m/s)")
plt.title("Velocidad vs Tiempo")
plt.grid()
plt.legend()

# Aceleración
plt.subplot(2,2,3)
plt.plot(t, a, label="Aceleración (m/s²)", color="green")
plt.scatter(t[np.argmax(a)], a_max, color="blue", label="Aceleración máx")
plt.xlabel("Tiempo (s)")
plt.ylabel("Aceleración (m/s²)")
plt.title("Aceleración vs Tiempo")
plt.grid()
plt.legend()

# Masa
plt.subplot(2,2,4)
plt.plot(t, m, label="Masa total (kg)", color="red")
plt.xlabel("Tiempo (s)")
plt.ylabel("Masa (kg)")
plt.title("Masa vs Tiempo")
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()

# ----- GRAFICA DE FUERZAS -----
plt.figure(figsize=(10,6))
plt.plot(t, F_thrust, label="Thrust (N)")
plt.plot(t, F_drag, label="Drag (N)")
plt.plot(t, F_weight, label="Peso (N)")
plt.xlabel("Tiempo (s)")
plt.ylabel("Fuerza (N)")
plt.title("Fuerzas vs Tiempo")
plt.grid()
plt.legend()
plt.show()
