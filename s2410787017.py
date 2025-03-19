"""Calculating tire forces using Pacejka Formula"""
import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy.constants import g, pi

# Libraries:
# numpy for array operations and math functions
# matplotlib for plotting graps
# argparse for argument parsing
# scipy.constants for math constants

# source for equations:
# Bakker, Egbert and Nyborg, Lars and Pacejka, Hans B.,
# „Tyre Modelling for Use in Vehicle Dynamics Studies“, SAE
# Transactions, 1987
# NOTE: equations with gamma(camber angle) removed as it is 0

# CONSTANTS
A1_FY = -22.1
A1_MZ = -2.72
A1_FX = -21.3
A2_FY = 1011
A2_MZ = -2.28
A2_FX = 1144
A3_FY = 1078
A3_MZ = -1.86
A3_FX = 49.6
A4_FY = 1.82
A4_MZ = -2.73
A4_FX = 226
A5_FY = 0.208
A5_MZ = 0.110
A5_FX = 0.069
A6_FY = 0.000
A6_MZ = -0.070
A6_FX = -0.006
A7_FY = -0.354
A7_MZ = 0.643
A7_FX = 0.056
A8_FY = 0.707
A8_MZ = -4.04
A8_FX = 0.486
C_SF = 1.30
C_BF = 1.65


def calculate_fz(mass):
  """Calculation of vertical force on single tire
  Args:
      mass: vehicle mass in kg
  returns:
      fz: vertical force in kN"""
  fz = (mass * g) / (4 * 1000)
  return fz

# Calculation for side force


def calculate_sf_fi(e_sf, alpha, b_sf):
  """Calculation for side force fi parameter
Args:
  e_sf: parameter for side force
  alpha: slip angle in radians
  b_sf: factor for side force
returns:
  fi: parameter fi for side force
"""
  return (1 - e_sf) * alpha + (e_sf / b_sf) * np.arctan(b_sf * alpha)


def calculate_sf_d(a1_fy, fz, a2_fy):
  """Calculation for side force parameter D
Args:
  a1_fy: coefficient
  fz: vertical force in kN
  a2_fy: coefficient
returns:
  float:parameter D
"""
  return a1_fy * (fz ** 2) + a2_fy * fz



def calculate_sf_b(a3_fy, fz, a4_fy, a5_fy, c_sf, d_sf):
  """Calculation for side force parameter B
Args:
  a3_fy: coefficient
  fz: vertical force in kN
  a4_fy: coefficient
  a5_fy: coefficient
  c_sf: coefficient
  d_sf: parameter D
returns:
  float: paramter B
"""
  return (a3_fy * np.sin(a4_fy * np.arctan(a5_fy * fz))) / (c_sf * d_sf)

# calculate e


def calculate_sf_e(a6_fy, fz, a7_fy, a8_fy):
  """Calculation for side force parameter E
Args:
  a6_fy: coefficient
  fz: vertical force in kN
  a7_fy: coefficient
  a5_fy: coefficient
  a8_fy: coefficient
returns:
  float: paramter E
"""
  return a6_fy * (fz ** 2) + a7_fy * fz + a8_fy



def calculate_fy(kappa, alpha, mass, c_sf, mu):
  """Calculation for lateral tire force Fy
Args:
  kappa: longitudinal slip ratio
  alpha: slip angle in radians
  mass: vehicle mass in kg
  c_sf: coefficient
  mu: friction coefficient
returns:
  Float: lateral force in N
"""
  fz = calculate_fz(mass)
  d_sf = calculate_sf_d(A1_FY, fz, A2_FY)
  e_sf = calculate_sf_e(A6_FY, fz, A7_FY, A8_FY)
  b_sf = calculate_sf_b(A3_FY, fz, A4_FY, A5_FY, c_sf, d_sf)
  fi_sf = calculate_sf_fi(e_sf, alpha, b_sf)
  fy_r = d_sf * np.sin(c_sf * np.arctan(b_sf * fi_sf))
  sigmay_x = -kappa / (1 + kappa)
  sigmay_y = -np.tan(alpha)
  sigmay = (sigmay_x ** 2 + sigmay_y ** 2) ** 0.5
  fy = -(sigmay_y / sigmay) * fy_r * mu * 1000

  return fy

# -----------------------------------------------------------------------------

# BRAKE FORCE


def calculate_bf_fi(e_bf, kappa, b_bf):
  """Calculation for brake force fi parameter
    Args:
      e_bf: parameter for side force
      kappa: longitudinal slip ratio
      b_sf: factor for side force
    returns:
      float: parameter fi for brake force
"""
  return (1 - e_bf) * kappa + (e_bf / b_bf) * np.arctan(b_bf * kappa)


def calculate_bf_d(a1_fx, fz, a2_fx):
  """Calculation for brake force parameter D
Args:
  a1_fx: coefficient
  fz: vertical force in kN
  a2_fx: coefficient
returns:
  float: parameter D
"""
  return a1_fx * (fz ** 2) + a2_fx * fz


def calculate_bf_b(a3_fx, fz, a4_fx, c_bf, d_bf, a5_fx):
  """Calculation for brake force parameter B
Args:
  a3_fx: coefficient
  fz: vertical force in kN
  a4_fx: coefficient
  c_bf: coefficient
  d_bf: parameter D
  a5_fx: coefficient
returns:
  float: parameter B
"""
  return ((a3_fx * (fz ** 2) + a4_fx * fz) /
          (c_bf * d_bf * np.exp(a5_fx * fz)))


def calculate_bf_e(a6_fx, fz, a7_fx, a8_fx):
  """Calculation for brake force parameter E
Args:
  a6_fx: coefficient
  fz: vertical force in kN
  a7_fx: coefficient
  a8_fx: coefficient
returns:
  float: parameter E
"""
  return a6_fx * (fz ** 2) + a7_fx * fz + a8_fx

# pylint: disable=too-many-positional-arguments


def calculate_fx(alpha, kappa, mass, c_bf, mu):
  """Calculation for longitudinal brake tire force Fx
Args:
  kappa: longitudinal slip ratio
  alpha: slip angle in radians
  mass: vehicle mass in kg
  c_bf: coefficient
  mu: friction coefficient
returns:
  Float: longitudinal force in N
"""
  fz = calculate_fz(mass)
  d_bf = calculate_bf_d(A1_FX, fz, A2_FX)
  e_bf = calculate_bf_e(A6_FX, fz, A7_FX, A8_FX)
  b_bf = calculate_bf_b(A3_FX, fz, A4_FX, c_bf, d_bf, A5_FX)
  fi_bf = calculate_bf_fi(e_bf, kappa, b_bf)

  fx_r = d_bf * np.sin(c_bf * np.arctan(b_bf * fi_bf))

  sigmax_x = -kappa / (1 + kappa)
  sigmax_y = -np.tan(alpha)
  sigmax = (sigmax_x ** 2 + sigmax_y ** 2) ** 0.5
  fx = -(sigmax_x / sigmax) * fx_r * mu * 1000
  return fx

# -----------------------------------------------------------------------------

def main():
  #Main function to parse argumenta and plot tire forces
  parser = argparse.ArgumentParser(
      description="Calculate and make a plot for tire forces")
  parser.add_argument("slip", type=float)
  parser.add_argument("weight", type=float)
  parser.add_argument("mu", type=float)
  args = parser.parse_args()

  #kappa = float(args.slip)
  kappa = np.linspace(0, 1, 100)

  alpha = args.slip * pi / 180
  # Slip angle(2 degree and convert to radians)
  mass = float(args.weight)
  # Vehicle mass in kilograms[kg]
  mu = float(args.mu)
  # mu without unit

  fy_values = [calculate_fy(k, alpha, mass, C_SF, mu) for k in kappa]
  fx_values = [calculate_fx(k, alpha, mass, C_BF, mu) for k in kappa]

  # make a graph
  plt.figure(figsize=(10, 6))
  plt.plot(kappa*100, fy_values, label="Side force - Fy [N]")
  plt.plot(kappa*100, fx_values, label="Brake force - Fx [N]")
  plt.title("Side force and brake force"
            " vs longitudinal slip "
            "at 2 degrees slip angle")
  plt.xlabel("Longitudinal slip - kappa [%]")
  plt.ylabel("Force [N]")
  plt.xlim(0, 100)
  # plt.ylim(0, 10000)
  plt.legend()
  plt.grid()

  # save graph
  plt.savefig("Tire_forces.png", format="png")


if __name__ == "__main__":
  main()
