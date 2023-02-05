# -*- coding: utf-8 -*-
"""
Electrical Waves Lissajous Data Analysis v2.0

Lukas Kostal, 4.2.2023, ICL
"""

import numpy as np
import scipy.constants as sc
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

# set numpy warning controller to ignore all errors
np.seterr(all="ignore")


# equation of line for fitting
def lin_fit(x, m, c):
    y = m * x + c
    return y

# log of equation of line
def log_fit(x, m, c):
    y = np.log(m * x + c) / np.log(10)
    return y

# equation of power curve for fitting
def pow_fit(x, a, b, c):
    y = a * x**b + c
    return y


# specify length over which measurements are taken in units of sections
l = 40

# inductance and capacitance in transmission line in H and F
L = 330e-6
L_err = 33e-6
C = 15e-9
C_err = 1.5e-9

# load data with units f(kHz), f_err(kHz), V1(V), V1_err(vV), V2(V), V2_err(mV)
f, f_err, V1, V1_err, V2, V2_err = np.loadtxt('Data/Dispersion.txt', unpack=True, skiprows=1)

# convert freq from kHz to Hz and error in voltage from mV to V
f *= 1e3
f_err *= 1e3
V1_err *= 1e-3
V2_err *= 1e-3 

# number of node being measured in phase for even and out of phase for odd
n = np.arange(0, len(f), 1)

# calculate angular freq and wavenumber
omg = 2 * np.pi * f
omg_err = 2 * np.pi * f_err
k = n * np.pi / l

# find gradient from linear part of diespersion relation
m = np.mean(np.diff(omg[:8]) / np.diff(k[:8]))

omg_t = 2 * np.sin(k / 2) / np.sqrt(L * C)
omg_t_err = omg_t * np.sqrt((L_err / L)**2 + (C_err / C)**2)

# plot dispersion relation
plt.errorbar(k, omg, yerr=omg_err, capsize=3, color='royalblue', label='measured')
plt.plot(k, omg_t, color='red', label='theoretical')
plt.plot(k, omg_t + omg_t_err, '--', color='red')
plt.plot(k, omg_t - omg_t_err, '--', color='red')
plt.plot(k, m * k, color='orange', label='linear')

# title and labels for plotting
plt.title('Lissajous Analysis Diesperion Relation')
plt.xlabel('wavenumber $k$ ($rad \: sec^{-1}$)')
plt.ylabel('angulr frequency $\omega$ ($rad \: s^{-1}$)')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.legend()

plt.savefig('Plots/Lissajous_disp.png', dpi=300, bbox_inches='tight')
plt.show()


# calculate the phase velocity and associated uncertainty
v_phase = (omg / k)
v_phase_err = (omg_err / k)

# calculate the group velocity and associated uncertainty
v_group = np.diff(omg) / np.diff(k)
v_group_err = np.sqrt(omg_err**2 + np.roll(omg_err, -1)**2)[:-1] / np.diff(k)

# curve fit a power curve onto the velocities
group_opt, group_cov = curve_fit(pow_fit, f[:-1], v_group, absolute_sigma=False)
phase_opt, phase_cov = curve_fit(pow_fit, f[1:], v_phase[1:], p0 = group_opt, absolute_sigma=False)

# calculate theoretical value for the phase velocity
v_phase_t = 1 / np.sqrt(L * C)
v_phase_t_err = 1/2 * np.sqrt(L_err**2 / (L**3 * C) + C_err**2 / (L * C**3))

# curve fitted parameters and expected uncertainty for phase velocity
phase_val = phase_opt
phase_err = np.sqrt(phase_cov / len(v_phase))

# curve fitted parameters and expected uncertainty group velocity
group_val = group_opt
group_err = np.sqrt(group_cov / len(v_group))

# print curve fit parameters
print()
print("Curve fit parameters y = ax^b + c")
print()
print("Phase velocity:")
print(f"a = {group_val[0]:.4g} \u00B1 {group_err[0, 0]:.4g} (4sf)")
print(f"b = {group_val[1]:.4g} \u00B1 {group_err[1, 1]:.4g} (4sf)")
print(f"c = {group_val[2]:.0f} \u00B1 {group_err[2, 2]:.0f} (0dp)")
print()
print("Group velocity:")
print(f"a = {phase_val[0]:.4g} \u00B1 {phase_err[0, 0]:.4g} (4sf)")
print(f"b = {phase_val[1]:.4g} \u00B1 {phase_err[1, 1]:.4g} (4sf)")
print(f"c = {phase_val[2]:.0f} \u00B1 {phase_err[2, 2]:.0f} (0dp)")
print()
print("Theoretical phase velocity:")
print(f"v_phase = {v_phase_t:.0f} \u00B1 {v_phase_t_err:.0f} sec s^-1 (0dp)")

# plot phase and group velocity against frequency
plt.errorbar(f, v_phase, xerr=f_err, yerr=v_phase_err, capsize=3, fmt='x', \
             color='royalblue', label='$v_{phase}$ measured')
plt.errorbar(f[:-1], v_group, xerr=f_err[:-1], yerr=v_group_err, capsize=3, fmt='x', \
             color='forestgreen', label='$v_{group}$ measured')

# plot theoretical phase velocity and fitted curves
# plt.plot(f, pow_fit(f, *phase_opt), color='hotpink', label='phase velocity fit')
# plt.plot(f, pow_fit(f, *group_opt), color='orange', label='group velocity fit')
plt.axhline(y=v_phase_t, color='red', label='$v_{phase}$ theoretical')
plt.axhline(y=v_phase_t + v_phase_t_err, linestyle='--', color='red')
plt.axhline(y=v_phase_t - v_phase_t_err, linestyle='--', color='red')

# title and labels for plotting
plt.title('Lissajous Analysis Phase and Group Velocity')
plt.xlabel('frequency $f$ ($Hz$)')
plt.ylabel('velocity $v$ ($sec \: s^{-1}$)')
plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
plt.legend()

plt.savefig('Plots/Lissajous_v.png', dpi=300, bbox_inches='tight')
plt.show()

# specify gain at which cutoff frequency is to be taken in dB
g_c = -3.01

# convert gain into amplitude ratio at which cutoff ratio is to be taken
A_c = 10**g_c

# calculate amplitude ratio and associated uncertainty
A = V2 / V1
A_err = A * np.sqrt((V1_err / V1)**2 + (V2_err / V2)**2)

# perform curve fit of line onto the amplitude ratio
A_opt, A_cov = curve_fit(lin_fit, f, A, absolute_sigma=False)

# parameter values from curve fit and expected uncertanties
m_A = A_opt[0]
c_A = A_opt[1]
m_A_err = np.sqrt(A_cov[0, 0] / len(A))
c_A_err = np.sqrt(A_cov[1, 1] / len(A))

# calculate cutoff frequency and expected uncertainty from amplitude ratio
fc_A = (A_c - c_A) / m_A
fc_A_err = np.sqrt(((A_c - c_A) / m_A**2)**2 * m_A_err**2 + (c_A_err / m_A)**2)

# calculate voltage gain and associated uncertainty in dB
g = np.log(V2 / V1) / np.log(10)
g_err = np.sqrt((V1_err / V1)**2 + (V2_err / V2)**2) / np.log(10)

# perform curve fit of line onto the voltage gain
g_opt, g_cov = curve_fit(log_fit, f, g)

# parameter values from curve fit and expected uncertanties
m_g = g_opt[0]
c_g = g_opt[1]
m_g_err = np.sqrt(g_cov[0, 0] / len(g))
c_g_err = np.sqrt(g_cov[1, 1] / len(g))

# calculate cutoff frequency and expected uncertainty from voltage gain
fc_g = (A_c - c_g) / m_g
fc_g_err = np.sqrt(((A_c - c_g) / m_g**2)**2 * m_g_err**2 + (c_g_err / m_g)**2)

# calculate the theoretical cutoff frequency in Hz
fc_t = 1 / np.sqrt(L * C) / np.pi
fc_t_err = 1 / 2 * fc_t*  np.sqrt((L_err / L)**2 + (C_err / C)**2)

# print the found cutoff frequencies and expected uncertanties
print()
print("Cutoff frequency calcualted from linear fit on A:")
print(f"fc_A = {fc_A:.0f} \u00B1 {fc_A_err:.0f} Hz (0dp)")
print()
print("Cutoff frequency calcualted from log fit on g:")
print(f"fc_g = {fc_g:.0f} \u00B1 {fc_g_err:.0f} Hz (0dp)")
print()
print("Theoretical cutoff frequency:")
print(f"fc_t = {fc_t:.0f} \u00B1 {fc_t_err:.0f} Hz (0dp)")

# array of frequencies for plotting
f_plot = np.linspace(0, 1.6e5, 1000)

# plot the amplitude ratio against frequency
plt.errorbar(f, A, xerr=f_err, yerr=A_err, fmt='x', capsize=3, color='royalblue', label='amplitude ratio')
plt.plot(f_plot, lin_fit(f_plot, *A_opt), color='orange', label='line fit')

plt.axhline(y=A_c, ls=':', color='red')
plt.axvline(x=fc_A, ls=':', color='red', label='cutoff frequency')

# title and labels for plotting
plt.title('Lissajous Analysis Amplitude Ratio')
plt.xlabel('frequency $f$ ($Hz$)')
plt.ylabel('amplitude ratio $A$ ($unitless$)')
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.legend(loc=6)

plt.savefig('Plots/Lissajous_A.png', dpi=300, bbox_inches='tight')
plt.show()

# plot voltage gain against frequency
plt.errorbar(f, g, xerr=f_err, yerr=g_err, capsize=3, fmt='x', color='royalblue', label='voltage gain')
plt.plot(f_plot, log_fit(f_plot, *g_opt), color='orange', label='log fit')

plt.axhline(y=g_c, ls=':', color='red')
plt.axvline(x=fc_g, ls=':', color='red', label='cutoff frequency')

# title and labels for plotting
plt.title('Lissajous Voltage Gain')
plt.xlabel('frequency $f$ ($Hz$)')
plt.ylabel('voltage gain $g$ ($dBV$)')
plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
plt.legend(loc=6)

plt.savefig('Plots/Lissajous_g.png', dpi=300, bbox_inches='tight')
plt.show()
