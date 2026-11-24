#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 22:26:23 2026

@author: frostadam8-ux

github repository: https://github.com/frostadam8-ux/WingProject.git
"""

import sys
import math
import numpy as np
import json
import matplotlib.pyplot as plt

DEBUG = True
AL0 = 0
ETAD = 1.0
INPUT_FILE = 'input.json'

# decided to go about loading this .json in a different way so the wing info can be an object
class WingInput:
    def __init__(self, filename):
        with open(filename, "r") as f:
            data = json.load(f)
        wing = data["wing"]
        planform = wing["planform"]
        washout = wing["washout"]
        aileron = wing["aileron"]
        # load planform
        self.planform_type = planform["type"].lower()
        self.aspect_ratio = planform["aspect_ratio"]
        self.taper_ratio = None
        self.planform_file = planform["filename"]
        # other stuff
        self.airfoil_lift_slope = wing["airfoil_lift_slope"]
        self.nodes_per_semispan = wing["nodes_per_semispan"]
        self.alpha_root_deg = data["condition"]["alpha_root[deg]"]
        if self.alpha_root_deg != "CL":
            self.alpha_root_rad = math.radians(self.alpha_root_deg)
        self.CL = data["condition"]["CL"]
        self.ail_deflection_deg = data["condition"]["aileron_deflection[deg]"]
        self.ail_deflection_rad = math.radians(self.ail_deflection_deg)
        self.pbar = data["condition"]["pbar"]
        # washout stuff
        self.washout_type = washout["distribution"].lower()
        self.washout_amount_deg = washout["amount[deg]"]
        if self.washout_amount_deg != 'optimum':
            self.washout_amount_rad = math.radians(self.washout_amount_deg)
        self.CL_design = washout["CL_design"]
        # aileron stuff
        self.ail_zb_start = aileron["begin[z/b]"]
        self.ail_zb_end = aileron["end[z/b]"]
        self.ail_cfc_start = aileron["begin[cf/c]"]
        self.ail_cfc_end = aileron["end[cf/c]"]
        self.ail_hinge_eff =aileron["hinge_efficiency"]
        # view stuff
        self.view_planform = data["view"]["planform"]
        self.view_washout_distribution = data["view"]["washout_distribution"]
        self.view_aileron_distribution = data["view"]["aileron_distribution"]
        self.view_CL_hat_distributions = data["view"]["CL_hat_distributions"]
        self.view_CL_tilde_distributions = data["view"]["CL_tilde_distributions"]

def get_nodes_from_geom(geometry):
    with open(geometry, "r") as f:
        info = []
        for line in f:
            try:
                info.append([float(coordinate) for coordinate in line.split()])
            except ValueError:
                continue
    # now make the info list a numpy array 
    nodes = np.array(info)
    mirror = nodes.copy()[1:,:]
    mirror[:,0] *= -1
    nodes = np.vstack((np.flip(mirror,axis=0), nodes))
    nodes[:,1] = np.maximum(nodes[:,1], 0.001)
    RA = 1/np.trapezoid(nodes[:,1],nodes[:,0])
    return nodes, RA

def calc_cb(wing, planform, zb):
    # get cb using tapered/elliptic numbers
    if wing.planform_type == 'tapered':
        cb = 2/(wing.aspect_ratio*(1+wing.taper_ratio)) * (1-(1-wing.taper_ratio)*np.abs(2*zb))
    elif wing.planform_type == 'elliptic':
        cb = 4/(np.pi*wing.aspect_ratio) * np.sqrt(1 - (2*zb)**2)
    elif wing.planform_type == 'file':
        cb = np.interp(zb, planform[:,0], planform[:,1])
    else:
        return 0
    return cb

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
    formatted = np.array2string(arr, separator=' ', max_line_width=np.inf, threshold=sys.maxsize)
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
        
def plot_planform(cb, zb, cfc, wing):
    fig, ax = plt.subplots(figsize=(10,4))
    # plot span
    ax.plot(zb,np.zeros((np.size(zb),1)), color='black')
    # plot le and te
    xle = 0.25*cb
    xte = -0.75*cb
    ax.plot(zb, xle, color='black')
    ax.plot(zb, xte, color='black')
    # plot ailerons
    for i in [-1,1]:
        x = np.array([wing.ail_zb_start,wing.ail_zb_start]) * i
        y = [np.interp(wing.ail_zb_start, zb, xte),hinge_start]
        ax.plot(x,y, color='black')
        x = np.array([wing.ail_zb_end,wing.ail_zb_end]) * i
        y = [np.interp(wing.ail_zb_end, zb, xte),hinge_end]
        ax.plot(x,y, color='black')
        x = np.array([wing.ail_zb_start,wing.ail_zb_end]) * i
        y = [hinge_start,hinge_end]
        ax.plot(x,y, color='black')
    # plot lifting lines
    ax.vlines(zb, xte, xle, colors='blue')
    # and red for the ailerons
    ax.vlines(zb, xte, cfc*cb+xte, colors='red')
    # set up final plot
    ax.set_xlabel('z/b')
    ax.set_ylabel('c/b')
    ax.set_aspect('equal')
    ax.set_title('Planform')
    plt.show()
    
def plot_washout(zb, omega):
    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(zb,omega, color='black')
    # set up final plot
    ax.set_xlabel('z/b')
    ax.set_ylabel('omega')
    ax.set_title('Washout Distribution')
    plt.show()
    
def plot_ail_distribution(zb, chi):
    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(zb,chi, color='black')
    # set up final plot
    ax.set_xlabel('z/b')
    ax.set_ylabel('chi')
    ax.set_title('Aileron Distribution')
    plt.show()
    
def clean(x, tol=1e-15):
    # helper function for final print to print zeros as zeros
    return 0.0 if abs(x) < tol else x
    
#%% main
if __name__ == "__main__":
    # grab object for the input .json
    wing = WingInput(INPUT_FILE)
    # get geometry if necessary
    if wing.planform_type == 'file':
        planform, RA = get_nodes_from_geom(wing.planform_file)
    else:
        planform, RA = None, wing.aspect_ratio
    # set up nodes (subtract one after doubling)
    N = wing.nodes_per_semispan*2 - 1
    theta = np.linspace(0, np.pi, N)
    j = np.arange(1, N+1)
    # get zb using cosine clustering
    zb = -0.5 + 0.5*(1-np.cos(theta))
    zb[np.abs(zb) < 1e-15] = 0
    # outsource cb to function so can use for aileron too
    cb = calc_cb(wing, planform, zb)
    cb[np.abs(cb) < 1e-15] = 0
    # get c, cinv, and an
    C = calc_c(N, wing.airfoil_lift_slope, cb, theta)
    C[np.abs(C) < 1e-15] = 0
    Cinv = np.linalg.inv(C)
    an = np.linalg.solve(C,np.full(N, 1))
    # get bn
    if wing.washout_type == 'linear':
        omega = np.abs(np.cos(theta))
    elif wing.washout_type == 'optimum':
        if wing.planform_type == 'elliptic':
            # elliptic optimum washout is none
            omega = np.zeros(N)
        else:
            omega = 1 - np.sin(theta)/(cb/calc_cb(wing, planform, 0.0))
    else:
        omega = np.zeros(N)
    # get bn
    bn = np.linalg.solve(C, omega)
    bn[np.abs(bn) < 1e-15] = 0
    # get aileron stuff starting with cf/c
    hinge_start = calc_cb(wing, planform, wing.ail_zb_start)*(wing.ail_cfc_start - 0.75)
    hinge_end = calc_cb(wing, planform, wing.ail_zb_end)*(wing.ail_cfc_end - 0.75)
    cfc = np.zeros(N)
    # do some numpy magic to handle the start/end cases
    on = (np.abs(zb) >= wing.ail_zb_start) & (np.abs(zb) <= wing.ail_zb_end)
    hinge = hinge_start + (hinge_end - hinge_start)*(np.abs(zb[on]) - wing.ail_zb_start)/(wing.ail_zb_end - wing.ail_zb_start)
    cfc[on] = hinge/cb[on] + 0.75
    # now for thetaf, epsfi, and epsf
    thetaf = np.arccos(2*cfc - 1)
    epsfi = 1 - (thetaf - np.sin(thetaf))/np.pi
    epsf = wing.ail_hinge_eff * ETAD * epsfi
    # now on to chi
    chi = np.zeros(N)
    chi[on] = epsf[on]
    chi[zb>0] = -chi[zb>0]
    # then that gives us cn
    cn = np.linalg.solve(C, chi)
    cn[np.abs(cn) < 1e-15] = 0
    # and we'll grab dn too
    dn = np.linalg.solve(C, np.cos(theta))
    #%% get coefficients
    kL = (1-(1+np.pi*RA/wing.airfoil_lift_slope)*an[0])/((1+np.pi*RA/wing.airfoil_lift_slope)*an[0])
    kD = np.sum([(n+1)*(an[n]/an[0])**2 for n in range(1,N)])
    epO = bn[0]/an[0]
    if bn[0]:
        kDL = 2 * (bn[0]/an[0]) * np.sum([(n+1)*(an[n]/an[0])*((bn[n]/bn[0])-(an[n]/an[0])) for n in range(1,N)])
        kDO = (bn[0]/an[0])**2 * np.sum([(n+1)*((bn[n]/bn[0])-(an[n]/an[0]))**2 for n in range(1,N)])
    else:
        kDL = kDO = 0
    CLa = wing.airfoil_lift_slope/((1+wing.airfoil_lift_slope/(np.pi*RA))*(1+kL))
    # deal with washout amount
    if wing.washout_amount_deg == 'optimum':
        Omega = kDL*wing.CL_design/(2*kDO*CLa)
    else:
        Omega = wing.washout_amount_rad
    # deal with roll rate
    Clda = -((np.pi*RA)/4)*cn[1]
    Clp = -((np.pi*RA)/4)*dn[1]
    pbar_steady = -(Clda/Clp)*wing.ail_deflection_rad
    if wing.pbar == 'steady':
        pbar = pbar_steady
    else:
        pbar = wing.pbar
    # get alpha_root
    if wing.alpha_root_deg == 'CL':
        alpha_root_rad = (wing.CL/CLa)+(epO*Omega)+AL0
    else:
        alpha_root_rad = math.radians(wing.alpha_root_deg)
    # then the final boss of fourier coeffs
    An = an*(alpha_root_rad-AL0)-(bn*Omega)+(cn*wing.ail_deflection_rad)+(dn*pbar)    
    # and then we get lift and drag!
    CL = np.pi*RA*An[0]
    CDi = (np.pi*RA*np.sum([(n+1)*An[n]**2 for n in range(0,N)])) - ((np.pi*RA*pbar)/2)*An[1]
    CLa = wing.airfoil_lift_slope/((1+wing.airfoil_lift_slope/(np.pi*RA))*(1+kL))
    sigma = np.sum([(n+1)*(An[n]/An[0])**2 for n in range(1,N)])
    es = 1/(1+sigma)
    Cl = (Clda*wing.ail_deflection_rad)+(Clp*pbar)
    Cn = ((CL/8)*(6*An[1]-pbar)) + (((np.pi*RA)/8)*(10*An[1]-pbar)*An[2]) + (((np.pi*RA)/4)*np.sum([(2*(n+1)-1)*An[n-1]*An[n] for n in range(3,N)]))
    # output the fourier stuff and coeffs
    write_to_file({
        "C": C,
        "C^{-1}": Cinv,
        "a_n": an.reshape(-1, 1),
        "b_n": bn.reshape(-1, 1),
        "c_n": cn.reshape(-1, 1),
        "d_n": dn.reshape(-1, 1),
        "A_n": An.reshape(-1, 1)
        }, 'Solution.txt')
    print(
          f'kappa_L = {clean(kL):.12g}\n'
          f'C_L,a = {clean(CLa):.12g}\n'
          f'epsilon_Omega = {clean(epO):.12g}\n'
          f'kappa_D = {clean(kD):.12g}\n'
          f'kappa_DL = {clean(kDL):.12g}\n'
          f'kappa_DOmega = {clean(kDO):.12g}\n\n'
          f'C_l,da = {clean(Clda):.12g}\n'
          f'C_l,pbar = {clean(Clp):.12g}\n\n'
          f'C_L = {clean(CL):.12g}\n'
          f'C_Di = {clean(CDi):.12g}\n\n'
          f'C_l = {clean(Cl):.12g}\n'
          f'C_n = {clean(Cn):.12g}\n'
          f'pbar_steady = {clean(pbar_steady):.12g}'
          )
    # generate plot of planform and washout
    if wing.view_planform:
        plot_planform(cb, zb, cfc, wing)
    if wing.view_washout_distribution:
        plot_washout(zb, omega)
    if wing.view_aileron_distribution:
        plot_ail_distribution(zb, chi)
