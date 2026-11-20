#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 22:26:23 2026

@author: admin
"""

import math
import numpy as np
import json
import matplotlib.pyplot as plt

DEBUG = True
AL0 = 0

# decided to go about loading this .json in a different way so the wing info can be an object
class WingInput:
    def __init__(self, filename):
        with open(filename, "r") as f:
            data = json.load(f)
        wing = data["wing"]
        planform = wing["planform"]
        # load planform
        self.planform_type = planform["type"].lower()
        self.aspect_ratio = planform["aspect_ratio"]
        self.taper_ratio = None
        if self.planform_type == "tapered":
            self.taper_ratio = planform["taper_ratio"]
        elif self.planform_type != "elliptic":
            # error if given type other than tapered or elliptic
            raise ValueError(f"Unknown planform type: {self.planform_type}")
        # other stuff
        self.airfoil_lift_slope = wing["airfoil_lift_slope"]
        self.nodes_per_semispan = wing["nodes_per_semispan"]
        self.alpha_root_deg = data["condition"]["alpha_root[deg]"]
        self.alpha_root_rad = math.radians(self.alpha_root_deg)
        self.view_planform = data["view"]["planform"]

def calc_c(N,sCla,c,theta):
    C = np.zeros((N,N))
    # put in the easy rows using some numpy magic
    row = np.arange(1,N+1) ** 2
    C[0,] = row
    row[1::2] *= -1
    C[N-1,] = row
    for i in range(N):
        if i == 0 or i == N-1:
            continue
        for j in range(1, N+1):
            C[i, j-1] = (4/(sCla*c[i]) + j/np.sin(theta[i])) * np.sin(j*theta[i])
    return C

def format_array(name, arr):
    arr = np.atleast_2d(arr)
    formatted = np.array2string(arr, separator=' ', max_line_width=np.inf)
    lines = [
        line.strip().lstrip('[').rstrip(']')
        for line in formatted.splitlines()
        if line.strip().lstrip('[').rstrip(']')
    ]
    lines[0] = f'{name} =\n{lines[0]}'
    return '\n'.join(lines)

def write_to_file(return_things,saveName):
    # make all of return things into strings
    formatted = []
    for name, item in return_things.items():
        if not isinstance(item, str):
            formatted.append(format_array(name, item))
    with open(saveName, mode='w', encoding='utf-8') as file:
        file.writelines(f"{item}\n\n" for item in formatted)
        
def plot_planform(cb, zb):
    fig, ax = plt.subplots(figsize=(10,4))
    # plot span
    ax.plot(zb,np.zeros((np.size(zb),1)), color='black')
    # plot le and te
    xle = 0.25*cb
    xte = -0.75*cb
    ax.plot(zb, xle, color='black')
    ax.plot(zb, xte, color='black')
    # plot lifting lines
    ax.vlines(zb, xte, xle, colors='blue')
    # set up final plot
    ax.set_xlabel('z/b')
    ax.set_ylabel('c/b')
    ax.set_aspect('equal')
    plt.show()
    
#%% W1 warmup
# grab object for the input .json
wing = WingInput("W1_input.json")
# set up nodes (subtract one after doubling)
N = wing.nodes_per_semispan*2 - 1
theta = np.linspace(0, np.pi, N)
j = np.arange(1, N+1)
# get zb using cosine clustering
zb = -0.5 + 0.5*(1-np.cos(theta))
zb[np.abs(zb) < 1e-15] = 0
# get cb using tapered/elliptic numbers
if wing.planform_type == 'tapered':
    cb = 2 / (wing.aspect_ratio*(1+wing.taper_ratio)) * (1-(1-wing.taper_ratio)*abs(np.cos(theta)))
elif wing.planform_type == 'elliptic':
    cb = 4*np.sin(theta) / (np.pi*wing.aspect_ratio)
    cb[np.abs(cb) < 1e-15] = 0
# get c, cinv, and an
C = calc_c(N, wing.airfoil_lift_slope, cb, theta)
C[np.abs(C) < 1e-15] = 0
Cinv = np.linalg.inv(C)
an = np.linalg.solve(C,np.full((N,1), 1))
# send all of these to a text file labeled "Solution.txt"
write_to_file({
    "C": C,
    "C^{-1}": Cinv,
    "a_n": an    
    }, 'Solution.txt')
# get kL, kD, es, CL,a, CL, and CDi then print
kL = (1-(1+np.pi*wing.aspect_ratio/wing.airfoil_lift_slope)*an[0,0])/((1+np.pi*wing.aspect_ratio/wing.airfoil_lift_slope)*an[0,0])
kD = np.sum([(n+1)*(an[n,0]/an[0,0])**2 for n in range(1,N)])
es = 1/(1+kD)
CLa = wing.airfoil_lift_slope/((1+wing.airfoil_lift_slope/(np.pi*wing.aspect_ratio))*(1+kL))
CL = CLa*(wing.alpha_root_rad-AL0)
CDi = CL**2/(np.pi*wing.aspect_ratio*es)
print(
      f'kappa_L = {kL}\n'
      f'kappa_D = {kD}\n'
      f'e_s = {es}\n'
      f'C_L,a = {CLa}\n\n'
      f'CL = {CL}\n'
      f'CDi = {CDi}\n'
      )
# generate plot of planform
if wing.view_planform:
    plot_planform(cb, zb)
