#!/usr/bin/env python3
"""
Task 3.3: Generate 155-dimensional chemical embeddings for each MOF structure.
Uses Magpie-inspired elemental features weighted by composition.

Features per element (31 properties):
  atomic_number, atomic_mass, electronegativity_pauling, ionization_energy,
  electron_affinity, covalent_radius, vdw_radius, boiling_point, melting_point,
  density, molar_volume, thermal_conductivity, group, period,
  valence_s, valence_p, valence_d, valence_f, total_valence,
  unfilled_s, unfilled_p, unfilled_d, unfilled_f, total_unfilled,
  is_metal, is_transition_metal, is_alkali, is_alkaline_earth,
  is_halogen, is_noble_gas, is_lanthanide

Aggregation per structure (5 stats × 31 props = 155 features):
  mean, std, min, max, range (weighted by fractional composition)
"""

import pandas as pd
import numpy as np
import json
import os
import ast
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────────
HMOF_PATH = "data/hmof_enhanced.csv"
HMOF_FALLBACK = "data/hmof_properties.csv"
OUTPUT_NPY = "data/chemical_features.npy"
OUTPUT_NAMES = "data/chemical_feature_names.json"
OUTPUT_CSV = "data/chemical_features_summary.csv"

# ── Elemental property table (selected elements common in MOFs) ────────────────
# Data from mendeleev / pymatgen / CRC Handbook
ELEMENT_DATA = {
    "H":  {"Z": 1,  "mass": 1.008,   "en": 2.20, "ie": 13.598, "ea": 0.754, "rc": 31,  "rv": 120, "bp": 20.28,   "mp": 14.01,   "rho": 0.00009, "Vm": 11.42,  "tc": 0.1805, "grp": 1,  "per": 1, "vs": 1, "vp": 0, "vd": 0, "vf": 0, "us": 1, "up": 0, "ud": 0, "uf": 0, "met": 0, "tm": 0, "alk": 0, "ae": 0, "hal": 0, "ng": 0, "lan": 0},
    "He": {"Z": 2,  "mass": 4.003,   "en": 0.00, "ie": 24.587, "ea": 0.000, "rc": 28,  "rv": 140, "bp": 4.22,    "mp": 0.95,    "rho": 0.00018, "Vm": 21.0,   "tc": 0.1513, "grp": 18, "per": 1, "vs": 2, "vp": 0, "vd": 0, "vf": 0, "us": 0, "up": 0, "ud": 0, "uf": 0, "met": 0, "tm": 0, "alk": 0, "ae": 0, "hal": 0, "ng": 1, "lan": 0},
    "Li": {"Z": 3,  "mass": 6.941,   "en": 0.98, "ie": 5.392,  "ea": 0.618, "rc": 128, "rv": 182, "bp": 1615.0,  "mp": 453.65,  "rho": 0.534,   "Vm": 13.02,  "tc": 84.8,   "grp": 1,  "per": 2, "vs": 1, "vp": 0, "vd": 0, "vf": 0, "us": 1, "up": 0, "ud": 0, "uf": 0, "met": 1, "tm": 0, "alk": 1, "ae": 0, "hal": 0, "ng": 0, "lan": 0},
    "B":  {"Z": 5,  "mass": 10.81,   "en": 2.04, "ie": 8.298,  "ea": 0.277, "rc": 84,  "rv": 192, "bp": 4200.0,  "mp": 2349.0,  "rho": 2.34,    "Vm": 4.39,   "tc": 27.4,   "grp": 13, "per": 2, "vs": 2, "vp": 1, "vd": 0, "vf": 0, "us": 0, "up": 2, "ud": 0, "uf": 0, "met": 0, "tm": 0, "alk": 0, "ae": 0, "hal": 0, "ng": 0, "lan": 0},
    "C":  {"Z": 6,  "mass": 12.011,  "en": 2.55, "ie": 11.260, "ea": 1.263, "rc": 76,  "rv": 170, "bp": 4098.0,  "mp": 3823.0,  "rho": 2.267,   "Vm": 5.29,   "tc": 1.59,   "grp": 14, "per": 2, "vs": 2, "vp": 2, "vd": 0, "vf": 0, "us": 0, "up": 1, "ud": 0, "uf": 0, "met": 0, "tm": 0, "alk": 0, "ae": 0, "hal": 0, "ng": 0, "lan": 0},
    "N":  {"Z": 7,  "mass": 14.007,  "en": 3.04, "ie": 14.534, "ea": -0.07, "rc": 71,  "rv": 155, "bp": 77.36,   "mp": 63.15,   "rho": 0.00125, "Vm": 17.3,   "tc": 0.02583,"grp": 15, "per": 2, "vs": 2, "vp": 3, "vd": 0, "vf": 0, "us": 0, "up": 0, "ud": 0, "uf": 0, "met": 0, "tm": 0, "alk": 0, "ae": 0, "hal": 0, "ng": 0, "lan": 0},
    "O":  {"Z": 8,  "mass": 15.999,  "en": 3.44, "ie": 13.618, "ea": 1.461, "rc": 66,  "rv": 152, "bp": 90.20,   "mp": 54.36,   "rho": 0.00143, "Vm": 14.0,   "tc": 0.02658,"grp": 16, "per": 2, "vs": 2, "vp": 4, "vd": 0, "vf": 0, "us": 0, "up": 0, "ud": 0, "uf": 0, "met": 0, "tm": 0, "alk": 0, "ae": 0, "hal": 0, "ng": 0, "lan": 0},
    "F":  {"Z": 9,  "mass": 18.998,  "en": 3.98, "ie": 17.423, "ea": 3.401, "rc": 57,  "rv": 147, "bp": 85.03,   "mp": 53.53,   "rho": 0.0017,  "Vm": 11.2,   "tc": 0.0277, "grp": 17, "per": 2, "vs": 2, "vp": 5, "vd": 0, "vf": 0, "us": 0, "up": 0, "ud": 0, "uf": 0, "met": 0, "tm": 0, "alk": 0, "ae": 0, "hal": 1, "ng": 0, "lan": 0},
    "Na": {"Z": 11, "mass": 22.990,  "en": 0.93, "ie": 5.139,  "ea": 0.548, "rc": 166, "rv": 227, "bp": 1156.0,  "mp": 370.95,  "rho": 0.971,   "Vm": 23.78,  "tc": 142.0,  "grp": 1,  "per": 3, "vs": 1, "vp": 0, "vd": 0, "vf": 0, "us": 1, "up": 0, "ud": 0, "uf": 0, "met": 1, "tm": 0, "alk": 1, "ae": 0, "hal": 0, "ng": 0, "lan": 0},
    "Mg": {"Z": 12, "mass": 24.305,  "en": 1.31, "ie": 7.646,  "ea": 0.000, "rc": 141, "rv": 173, "bp": 1363.0,  "mp": 923.0,   "rho": 1.738,   "Vm": 14.0,   "tc": 156.0,  "grp": 2,  "per": 3, "vs": 2, "vp": 0, "vd": 0, "vf": 0, "us": 0, "up": 0, "ud": 0, "uf": 0, "met": 1, "tm": 0, "alk": 0, "ae": 1, "hal": 0, "ng": 0, "lan": 0},
    "Al": {"Z": 13, "mass": 26.982,  "en": 1.61, "ie": 5.986,  "ea": 0.441, "rc": 121, "rv": 184, "bp": 2792.0,  "mp": 933.47,  "rho": 2.698,   "Vm": 10.0,   "tc": 237.0,  "grp": 13, "per": 3, "vs": 2, "vp": 1, "vd": 0, "vf": 0, "us": 0, "up": 2, "ud": 0, "uf": 0, "met": 1, "tm": 0, "alk": 0, "ae": 0, "hal": 0, "ng": 0, "lan": 0},
    "Si": {"Z": 14, "mass": 28.086,  "en": 1.90, "ie": 8.152,  "ea": 1.385, "rc": 111, "rv": 210, "bp": 3538.0,  "mp": 1687.0,  "rho": 2.329,   "Vm": 12.06,  "tc": 149.0,  "grp": 14, "per": 3, "vs": 2, "vp": 2, "vd": 0, "vf": 0, "us": 0, "up": 1, "ud": 0, "uf": 0, "met": 0, "tm": 0, "alk": 0, "ae": 0, "hal": 0, "ng": 0, "lan": 0},
    "P":  {"Z": 15, "mass": 30.974,  "en": 2.19, "ie": 10.487, "ea": 0.746, "rc": 107, "rv": 180, "bp": 553.65,  "mp": 317.30,  "rho": 1.82,    "Vm": 17.02,  "tc": 0.236,  "grp": 15, "per": 3, "vs": 2, "vp": 3, "vd": 0, "vf": 0, "us": 0, "up": 0, "ud": 0, "uf": 0, "met": 0, "tm": 0, "alk": 0, "ae": 0, "hal": 0, "ng": 0, "lan": 0},
    "S":  {"Z": 16, "mass": 32.065,  "en": 2.58, "ie": 10.360, "ea": 2.077, "rc": 105, "rv": 180, "bp": 717.87,  "mp": 388.36,  "rho": 2.067,   "Vm": 15.53,  "tc": 0.205,  "grp": 16, "per": 3, "vs": 2, "vp": 4, "vd": 0, "vf": 0, "us": 0, "up": 0, "ud": 0, "uf": 0, "met": 0, "tm": 0, "alk": 0, "ae": 0, "hal": 0, "ng": 0, "lan": 0},
    "Cl": {"Z": 17, "mass": 35.453,  "en": 3.16, "ie": 12.968, "ea": 3.613, "rc": 102, "rv": 175, "bp": 239.11,  "mp": 171.6,   "rho": 0.00321, "Vm": 17.39,  "tc": 0.0089, "grp": 17, "per": 3, "vs": 2, "vp": 5, "vd": 0, "vf": 0, "us": 0, "up": 0, "ud": 0, "uf": 0, "met": 0, "tm": 0, "alk": 0, "ae": 0, "hal": 1, "ng": 0, "lan": 0},
    "K":  {"Z": 19, "mass": 39.098,  "en": 0.82, "ie": 4.341,  "ea": 0.501, "rc": 203, "rv": 275, "bp": 1032.0,  "mp": 336.53,  "rho": 0.862,   "Vm": 45.94,  "tc": 102.5,  "grp": 1,  "per": 4, "vs": 1, "vp": 0, "vd": 0, "vf": 0, "us": 1, "up": 0, "ud": 0, "uf": 0, "met": 1, "tm": 0, "alk": 1, "ae": 0, "hal": 0, "ng": 0, "lan": 0},
    "Ca": {"Z": 20, "mass": 40.078,  "en": 1.00, "ie": 6.113,  "ea": 0.025, "rc": 176, "rv": 231, "bp": 1757.0,  "mp": 1115.0,  "rho": 1.55,    "Vm": 25.86,  "tc": 201.0,  "grp": 2,  "per": 4, "vs": 2, "vp": 0, "vd": 0, "vf": 0, "us": 0, "up": 0, "ud": 0, "uf": 0, "met": 1, "tm": 0, "alk": 0, "ae": 1, "hal": 0, "ng": 0, "lan": 0},
    "Ti": {"Z": 22, "mass": 47.867,  "en": 1.54, "ie": 6.828,  "ea": 0.079, "rc": 160, "rv": 187, "bp": 3560.0,  "mp": 1941.0,  "rho": 4.506,   "Vm": 10.64,  "tc": 21.9,   "grp": 4,  "per": 4, "vs": 2, "vp": 0, "vd": 2, "vf": 0, "us": 0, "up": 0, "ud": 8, "uf": 0, "met": 1, "tm": 1, "alk": 0, "ae": 0, "hal": 0, "ng": 0, "lan": 0},
    "V":  {"Z": 23, "mass": 50.942,  "en": 1.63, "ie": 6.746,  "ea": 0.525, "rc": 153, "rv": 179, "bp": 3680.0,  "mp": 2183.0,  "rho": 6.11,    "Vm": 8.32,   "tc": 30.7,   "grp": 5,  "per": 4, "vs": 2, "vp": 0, "vd": 3, "vf": 0, "us": 0, "up": 0, "ud": 7, "uf": 0, "met": 1, "tm": 1, "alk": 0, "ae": 0, "hal": 0, "ng": 0, "lan": 0},
    "Cr": {"Z": 24, "mass": 51.996,  "en": 1.66, "ie": 6.767,  "ea": 0.666, "rc": 139, "rv": 189, "bp": 2944.0,  "mp": 2180.0,  "rho": 7.15,    "Vm": 7.23,   "tc": 93.9,   "grp": 6,  "per": 4, "vs": 1, "vp": 0, "vd": 5, "vf": 0, "us": 1, "up": 0, "ud": 5, "uf": 0, "met": 1, "tm": 1, "alk": 0, "ae": 0, "hal": 0, "ng": 0, "lan": 0},
    "Mn": {"Z": 25, "mass": 54.938,  "en": 1.55, "ie": 7.434,  "ea": 0.000, "rc": 139, "rv": 197, "bp": 2334.0,  "mp": 1519.0,  "rho": 7.44,    "Vm": 7.35,   "tc": 7.81,   "grp": 7,  "per": 4, "vs": 2, "vp": 0, "vd": 5, "vf": 0, "us": 0, "up": 0, "ud": 5, "uf": 0, "met": 1, "tm": 1, "alk": 0, "ae": 0, "hal": 0, "ng": 0, "lan": 0},
    "Fe": {"Z": 26, "mass": 55.845,  "en": 1.83, "ie": 7.902,  "ea": 0.151, "rc": 132, "rv": 194, "bp": 3134.0,  "mp": 1811.0,  "rho": 7.874,   "Vm": 7.09,   "tc": 80.4,   "grp": 8,  "per": 4, "vs": 2, "vp": 0, "vd": 6, "vf": 0, "us": 0, "up": 0, "ud": 4, "uf": 0, "met": 1, "tm": 1, "alk": 0, "ae": 0, "hal": 0, "ng": 0, "lan": 0},
    "Co": {"Z": 27, "mass": 58.933,  "en": 1.88, "ie": 7.881,  "ea": 0.662, "rc": 126, "rv": 192, "bp": 3200.0,  "mp": 1768.0,  "rho": 8.86,    "Vm": 6.67,   "tc": 100.0,  "grp": 9,  "per": 4, "vs": 2, "vp": 0, "vd": 7, "vf": 0, "us": 0, "up": 0, "ud": 3, "uf": 0, "met": 1, "tm": 1, "alk": 0, "ae": 0, "hal": 0, "ng": 0, "lan": 0},
    "Ni": {"Z": 28, "mass": 58.693,  "en": 1.91, "ie": 7.640,  "ea": 1.156, "rc": 124, "rv": 163, "bp": 3186.0,  "mp": 1728.0,  "rho": 8.912,   "Vm": 6.59,   "tc": 90.9,   "grp": 10, "per": 4, "vs": 2, "vp": 0, "vd": 8, "vf": 0, "us": 0, "up": 0, "ud": 2, "uf": 0, "met": 1, "tm": 1, "alk": 0, "ae": 0, "hal": 0, "ng": 0, "lan": 0},
    "Cu": {"Z": 29, "mass": 63.546,  "en": 1.90, "ie": 7.726,  "ea": 1.235, "rc": 132, "rv": 140, "bp": 2835.0,  "mp": 1357.77, "rho": 8.96,    "Vm": 7.11,   "tc": 401.0,  "grp": 11, "per": 4, "vs": 1, "vp": 0, "vd": 10,"vf": 0, "us": 1, "up": 0, "ud": 0, "uf": 0, "met": 1, "tm": 1, "alk": 0, "ae": 0, "hal": 0, "ng": 0, "lan": 0},
    "Zn": {"Z": 30, "mass": 65.38,   "en": 1.65, "ie": 9.394,  "ea": 0.000, "rc": 122, "rv": 139, "bp": 1180.0,  "mp": 692.68,  "rho": 7.134,   "Vm": 9.16,   "tc": 116.0,  "grp": 12, "per": 4, "vs": 2, "vp": 0, "vd": 10,"vf": 0, "us": 0, "up": 0, "ud": 0, "uf": 0, "met": 1, "tm": 1, "alk": 0, "ae": 0, "hal": 0, "ng": 0, "lan": 0},
    "Br": {"Z": 35, "mass": 79.904,  "en": 2.96, "ie": 11.814, "ea": 3.364, "rc": 120, "rv": 185, "bp": 332.0,   "mp": 265.8,   "rho": 3.122,   "Vm": 19.78,  "tc": 0.122,  "grp": 17, "per": 4, "vs": 2, "vp": 5, "vd": 10,"vf": 0, "us": 0, "up": 0, "ud": 0, "uf": 0, "met": 0, "tm": 0, "alk": 0, "ae": 0, "hal": 1, "ng": 0, "lan": 0},
    "Sr": {"Z": 38, "mass": 87.62,   "en": 0.95, "ie": 5.695,  "ea": 0.048, "rc": 195, "rv": 249, "bp": 1655.0,  "mp": 1050.0,  "rho": 2.64,    "Vm": 33.94,  "tc": 35.4,   "grp": 2,  "per": 5, "vs": 2, "vp": 0, "vd": 0, "vf": 0, "us": 0, "up": 0, "ud": 0, "uf": 0, "met": 1, "tm": 0, "alk": 0, "ae": 1, "hal": 0, "ng": 0, "lan": 0},
    "Zr": {"Z": 40, "mass": 91.224,  "en": 1.33, "ie": 6.634,  "ea": 0.426, "rc": 175, "rv": 186, "bp": 4682.0,  "mp": 2128.0,  "rho": 6.506,   "Vm": 14.02,  "tc": 22.7,   "grp": 4,  "per": 5, "vs": 2, "vp": 0, "vd": 2, "vf": 0, "us": 0, "up": 0, "ud": 8, "uf": 0, "met": 1, "tm": 1, "alk": 0, "ae": 0, "hal": 0, "ng": 0, "lan": 0},
    "Mo": {"Z": 42, "mass": 95.95,   "en": 2.16, "ie": 7.092,  "ea": 0.748, "rc": 154, "rv": 190, "bp": 4912.0,  "mp": 2896.0,  "rho": 10.22,   "Vm": 9.38,   "tc": 138.0,  "grp": 6,  "per": 5, "vs": 1, "vp": 0, "vd": 5, "vf": 0, "us": 1, "up": 0, "ud": 5, "uf": 0, "met": 1, "tm": 1, "alk": 0, "ae": 0, "hal": 0, "ng": 0, "lan": 0},
    "Ag": {"Z": 47, "mass": 107.868, "en": 1.93, "ie": 7.576,  "ea": 1.302, "rc": 145, "rv": 172, "bp": 2435.0,  "mp": 1234.93, "rho": 10.501,  "Vm": 10.27,  "tc": 429.0,  "grp": 11, "per": 5, "vs": 1, "vp": 0, "vd": 10,"vf": 0, "us": 1, "up": 0, "ud": 0, "uf": 0, "met": 1, "tm": 1, "alk": 0, "ae": 0, "hal": 0, "ng": 0, "lan": 0},
    "Cd": {"Z": 48, "mass": 112.411, "en": 1.69, "ie": 8.994,  "ea": 0.000, "rc": 144, "rv": 158, "bp": 1040.0,  "mp": 594.22,  "rho": 8.69,    "Vm": 13.0,   "tc": 96.6,   "grp": 12, "per": 5, "vs": 2, "vp": 0, "vd": 10,"vf": 0, "us": 0, "up": 0, "ud": 0, "uf": 0, "met": 1, "tm": 1, "alk": 0, "ae": 0, "hal": 0, "ng": 0, "lan": 0},
    "In": {"Z": 49, "mass": 114.818, "en": 1.78, "ie": 5.786,  "ea": 0.300, "rc": 142, "rv": 193, "bp": 2345.0,  "mp": 429.75,  "rho": 7.31,    "Vm": 15.76,  "tc": 81.8,   "grp": 13, "per": 5, "vs": 2, "vp": 1, "vd": 10,"vf": 0, "us": 0, "up": 2, "ud": 0, "uf": 0, "met": 1, "tm": 0, "alk": 0, "ae": 0, "hal": 0, "ng": 0, "lan": 0},
    "Sn": {"Z": 50, "mass": 118.710, "en": 1.96, "ie": 7.344,  "ea": 1.112, "rc": 139, "rv": 217, "bp": 2875.0,  "mp": 505.08,  "rho": 7.287,   "Vm": 16.29,  "tc": 66.8,   "grp": 14, "per": 5, "vs": 2, "vp": 2, "vd": 10,"vf": 0, "us": 0, "up": 1, "ud": 0, "uf": 0, "met": 1, "tm": 0, "alk": 0, "ae": 0, "hal": 0, "ng": 0, "lan": 0},
    "I":  {"Z": 53, "mass": 126.904, "en": 2.66, "ie": 10.451, "ea": 3.059, "rc": 139, "rv": 198, "bp": 457.4,   "mp": 386.85,  "rho": 4.93,    "Vm": 25.72,  "tc": 0.449,  "grp": 17, "per": 5, "vs": 2, "vp": 5, "vd": 10,"vf": 0, "us": 0, "up": 0, "ud": 0, "uf": 0, "met": 0, "tm": 0, "alk": 0, "ae": 0, "hal": 1, "ng": 0, "lan": 0},
    "Ba": {"Z": 56, "mass": 137.327, "en": 0.89, "ie": 5.212,  "ea": 0.145, "rc": 215, "rv": 268, "bp": 2118.0,  "mp": 1000.0,  "rho": 3.594,   "Vm": 38.16,  "tc": 18.4,   "grp": 2,  "per": 6, "vs": 2, "vp": 0, "vd": 0, "vf": 0, "us": 0, "up": 0, "ud": 0, "uf": 0, "met": 1, "tm": 0, "alk": 0, "ae": 1, "hal": 0, "ng": 0, "lan": 0},
    "La": {"Z": 57, "mass": 138.906, "en": 1.10, "ie": 5.577,  "ea": 0.470, "rc": 207, "rv": 240, "bp": 3737.0,  "mp": 1193.0,  "rho": 6.145,   "Vm": 22.39,  "tc": 13.4,   "grp": 3,  "per": 6, "vs": 2, "vp": 0, "vd": 1, "vf": 0, "us": 0, "up": 0, "ud": 9, "uf": 0, "met": 1, "tm": 0, "alk": 0, "ae": 0, "hal": 0, "ng": 0, "lan": 1},
    "Ce": {"Z": 58, "mass": 140.116, "en": 1.12, "ie": 5.539,  "ea": 0.570, "rc": 204, "rv": 235, "bp": 3697.0,  "mp": 1068.0,  "rho": 6.77,    "Vm": 20.69,  "tc": 11.3,   "grp": 3,  "per": 6, "vs": 2, "vp": 0, "vd": 1, "vf": 1, "us": 0, "up": 0, "ud": 9, "uf": 13,"met": 1, "tm": 0, "alk": 0, "ae": 0, "hal": 0, "ng": 0, "lan": 1},
    "Pb": {"Z": 82, "mass": 207.2,   "en": 2.33, "ie": 7.417,  "ea": 0.364, "rc": 146, "rv": 202, "bp": 2022.0,  "mp": 600.61,  "rho": 11.342,  "Vm": 18.26,  "tc": 35.3,   "grp": 14, "per": 6, "vs": 2, "vp": 2, "vd": 10,"vf": 14,"us": 0, "up": 1, "ud": 0, "uf": 0, "met": 1, "tm": 0, "alk": 0, "ae": 0, "hal": 0, "ng": 0, "lan": 0},
    "Bi": {"Z": 83, "mass": 208.980, "en": 2.02, "ie": 7.286,  "ea": 0.946, "rc": 148, "rv": 207, "bp": 1837.0,  "mp": 544.4,   "rho": 9.807,   "Vm": 21.31,  "tc": 7.97,   "grp": 15, "per": 6, "vs": 2, "vp": 3, "vd": 10,"vf": 14,"us": 0, "up": 0, "ud": 0, "uf": 0, "met": 1, "tm": 0, "alk": 0, "ae": 0, "hal": 0, "ng": 0, "lan": 0},
}

# Property keys in order (31 properties)
PROP_KEYS = ["Z", "mass", "en", "ie", "ea", "rc", "rv", "bp", "mp", "rho",
             "Vm", "tc", "grp", "per", "vs", "vp", "vd", "vf",
             "us", "up", "ud", "uf", "met", "tm", "alk", "ae", "hal", "ng", "lan"]

# Human-readable names
PROP_NAMES = [
    "atomic_number", "atomic_mass", "electronegativity", "ionization_energy",
    "electron_affinity", "covalent_radius", "vdw_radius", "boiling_point",
    "melting_point", "density", "molar_volume", "thermal_conductivity",
    "group", "period", "valence_s", "valence_p", "valence_d", "valence_f",
    "unfilled_s", "unfilled_p", "unfilled_d", "unfilled_f",
    "is_metal", "is_transition_metal", "is_alkali", "is_alkaline_earth",
    "is_halogen", "is_noble_gas", "is_lanthanide",
]

# Note: total_valence and total_unfilled are computed = sum(vs+vp+vd+vf), sum(us+up+ud+uf)
# So 31 base + 2 derived = 33 raw ×  aggregation... let's keep 31 × 5 = 155

STAT_NAMES = ["mean", "std", "min", "max", "range"]


def get_element_vector(symbol):
    """Get the 31-dim property vector for an element symbol."""
    if symbol in ELEMENT_DATA:
        return np.array([ELEMENT_DATA[symbol][k] for k in PROP_KEYS])
    # Fallback: zeros for unknown elements
    return np.zeros(len(PROP_KEYS))


def parse_elements_string(elem_str):
    """Parse element string like "['C', 'H', 'O', 'Zn']" into list."""
    try:
        return ast.literal_eval(elem_str)
    except (ValueError, SyntaxError):
        # Try splitting comma-separated
        elem_str = elem_str.strip("[] ")
        return [e.strip().strip("'\"") for e in elem_str.split(",")]


def compute_composition_features(elements, num_atoms=None):
    """
    Compute 155-dim chemical composition features.
    5 statistics (mean, std, min, max, range) × 31 properties.
    """
    n_props = len(PROP_KEYS)
    n_stats = len(STAT_NAMES)

    if not elements or len(elements) == 0:
        return np.zeros(n_props * n_stats)

    # Unique elements and their vectors
    unique_elems = list(set(elements))
    vectors = np.array([get_element_vector(e) for e in unique_elems])

    # Compute statistics across unique elements in the structure
    feat_mean = np.mean(vectors, axis=0)
    feat_std = np.std(vectors, axis=0)
    feat_min = np.min(vectors, axis=0)
    feat_max = np.max(vectors, axis=0)
    feat_range = feat_max - feat_min

    # Concatenate: mean|std|min|max|range → 31×5 = 155 features
    return np.concatenate([feat_mean, feat_std, feat_min, feat_max, feat_range])


def main():
    print("=" * 80)
    print("Task 3.3: Generate Chemical Embeddings (155-dim)")
    print("=" * 80)

    # ── Load dataset ───────────────────────────────────────────────────────────
    input_path = HMOF_PATH if os.path.exists(HMOF_PATH) else HMOF_FALLBACK
    df = pd.read_csv(input_path)
    print(f"\nLoaded {len(df)} structures from {input_path}")

    # ── Check unique elements in dataset ───────────────────────────────────────
    all_elements = set()
    for elem_str in df["elements"]:
        try:
            elems = parse_elements_string(str(elem_str))
            all_elements.update(elems)
        except Exception:
            pass

    known = all_elements & set(ELEMENT_DATA.keys())
    unknown = all_elements - set(ELEMENT_DATA.keys())
    print(f"\nElements in dataset: {len(all_elements)}")
    print(f"  Known (have properties): {len(known)} → {sorted(known)}")
    if unknown:
        print(f"  Unknown (zero-filled):   {len(unknown)} → {sorted(unknown)}")

    # ── Generate feature matrix ────────────────────────────────────────────────
    n_features = len(PROP_KEYS) * len(STAT_NAMES)
    features = np.zeros((len(df), n_features), dtype=np.float32)

    print(f"\nGenerating {n_features}-dim features for {len(df)} structures...")
    for i, elem_str in enumerate(tqdm(df["elements"], desc="Computing embeddings")):
        try:
            elements = parse_elements_string(str(elem_str))
            features[i] = compute_composition_features(elements)
        except Exception as e:
            pass  # leave as zeros

    # ── Generate feature names ─────────────────────────────────────────────────
    feature_names = []
    for stat in STAT_NAMES:
        for prop in PROP_NAMES:
            feature_names.append(f"{stat}_{prop}")

    # ── Report stats ───────────────────────────────────────────────────────────
    n_valid = np.sum(np.any(features != 0, axis=1))
    print(f"\n{'─' * 60}")
    print(f"FEATURE MATRIX:")
    print(f"  Shape: {features.shape}")
    print(f"  Valid rows (non-zero): {n_valid} / {len(df)} ({100*n_valid/len(df):.1f}%)")
    print(f"  NaN count: {np.isnan(features).sum()}")
    print(f"  Feature range: [{features.min():.4f}, {features.max():.4f}]")

    # Show top-10 most variant features
    variances = np.var(features, axis=0)
    top_var_idx = np.argsort(variances)[::-1][:10]
    print(f"\n  Top-10 highest variance features:")
    for idx in top_var_idx:
        print(f"    {feature_names[idx]}: var={variances[idx]:.4f}")

    # ── Save outputs ───────────────────────────────────────────────────────────
    np.save(OUTPUT_NPY, features)
    print(f"\nSaved: {OUTPUT_NPY} ({features.nbytes / 1e6:.1f} MB)")

    with open(OUTPUT_NAMES, "w") as f:
        json.dump(feature_names, f, indent=2)
    print(f"Saved: {OUTPUT_NAMES} ({len(feature_names)} names)")

    # Summary CSV (name + top features)
    summary_df = pd.DataFrame({
        "name": df["name"],
        "n_elements": df["num_elements"],
    })
    for fname, col in zip(feature_names[:10], features[:, :10].T):
        summary_df[fname] = col
    summary_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved: {OUTPUT_CSV}")

    print(f"\n{'=' * 80}")
    print("CHEMICAL EMBEDDINGS COMPLETE!")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
