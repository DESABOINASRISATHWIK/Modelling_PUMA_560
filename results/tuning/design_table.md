# PD gain design (joints 1-3 from wn/zeta; wrist joints damping dominated)

| joint | I_min | I_max | wn [rad/s] | Kp design | Kd design | Kd cap I_min/dt | Kp used | Kd used |
|---|---|---|---|---|---|---|---|---|
| 1 | 1.57 | 4.25 | 12 | 611.9 | 101 | 786 | 600 | 100 |
| 2 | 2 | 2.88 | 14 | 565.0 | 79.7 | 999 | 550 | 80 |
| 3 | 0.359 | 0.362 | 25 | 226.1 | 17.1 | 179 | 225 | 17 |
| 4 | 0.00164 | 0.00184 | n/a (damping-dominated) | - | 0.01 | 0.82 | 20 | 0.5 |
| 5 | 0.000642 | 0.000642 | n/a (damping-dominated) | - | 0.01 | 0.321 | 20 | 0.2 |
| 6 | 4e-05 | 4e-05 | n/a (damping-dominated) | - | 0.01 | 0.02 | 20 | 0.01 |

Simulator viscous damping b = 1 N m s/rad, control period 2 ms.
Wrist Kp is set from the closed-loop time constant (b + Kd)/Kp ~ 50 ms.
