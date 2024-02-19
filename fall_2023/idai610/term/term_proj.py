#!/usr/bin/env python
"""
-------------------------
PREDICTING DISENGAGEMENT
-------------------------
Disengagement - "a deactivation of the autonomous mode when a failure of the
autonomous technology is detected or when the safe operation of the vehicle
requires that the autonomous vehicle test driver disengage the autonomous mode
and take immediate manual. control of the vehicle."

This dataset of disengagement reports can highlight where the technology is still struggling.
--------------------------------------------------------------------------------------------
"The data files below contain the disengagements and autonomous miles traveled for permit holders who reported testing on California’s public roads between December 1, 2018 and November 30, 2019. Separate data files contain information on companies who received their permit in 2018 and are reporting testing activity in California for the first time (beyond the normal 12-month cycle)." Each report includes:

Manufacturer
Permit Number
DATE
VIN NUMBER
VEHICLE IS CAPABLE OF OPERATING WITHOUT A DRIVER(Yes or No)
DRIVER PRESENT(Yes or No)
DISENGAGEMENT INITIATED BY(AV System, Test Driver, Remote Operator, or Passenger)
DISENGAGEMENTLOCATION (Interstate, Freeway, Highway, Rural Road, Street, or Parking Facility)
DESCRIPTION OF FACTS CAUSING DISENGAGEMENT
"""
import pandas as pd
import numpy as np

df = pd.read_csv(r"term_data.csv")
df.fillna(0, inplace=True)


