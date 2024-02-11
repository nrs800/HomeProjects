#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 13:03:40 2024

@author: nathanaelseay
"""
import pandas as pd

from Anomaly_Shaper import Shaper

bearing = 4

wave_data = pd.DataFrame()

wave_data = Shaper(bearing)