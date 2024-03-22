from __future__ import division, unicode_literals, print_function # for compatibility with Python 2 and 3
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
from pandas import DataFrame, Series # for convenience

import pims
import trackpy as tp

@pims.pipeline
def gray(image):
    """
    Converts an image coloring to grayscale.
    """
    return image[:, :, 0] # Take just the green channel

"""
Start of running code.
"""
# STEP 1: Read the data
frames = gray(pims.open('sample_data/bulk_water/*.png'))

# plt.imshow(frames[0])
# plt.show()

# STEP 2: Locate features
# Estimate 11 pixels for the size of the features. 
# The size must be an odd int, and it's better to guess too large.
# The algorithm looks for bright features. Since the features in
# this image set are dark, we set invert=True.
# Locate returns a spreadsheet-like object called a DataFrame. 
# f = tp.locate(frames[0], 11, invert=True) 

# Print first few rows of DataFrame
# print(f.head())

# Specify minmass and threshold parameters to improve readability of noisy images
# There are more options to optimize feature-finding in the documentation
# f = tp.locate(frames[0], 11, invert=True, minmass=20) 

# Locate features in the first 300 frames
# NOTE: This isn't working. Might need to iterate over frames in a for loop.
# Can we fork the batch commands onto separate threads? 
# -> FIXED: addes processes=1 to disable multi-processes
f = tp.batch(frames[:300], 11, minmass=20, invert=True, processes=1)

# Turn off progress reports for best performance
tp.quiet()

# Draws red circles around the features
# tp.annotate(f, frames[0])

# Refine parameters to eliminate spurious features
# fig, ax = plt.subplots()
# ax.hist(f['mass'], bins=20)
# ax.set(xlabel='mass', ylabel='count') # mass is total brightness
# plt.show()

# Check that the decimal part of x and/or y positions are evenly distributed
# tp.subpx_bias(f)
# plt.show()

# STEP 3: Link features into particle trajectories
# maximum displacement = 5 (the farthest a particle can travel between frames)
# memory keeps track of disappeared particles and maintains their id for 3 frames
# Numbers each particle
t = tp.link(f, 5, memory=3)
# print(t.head())

# filter_stubs keeps only trajectories that last for a given number of frames
# trajectories that only last a few frames are never useful
t1 = tp.filter_stubs(t, 25)
print('Before:', t['particle'].nunique())
print('After:', t1['particle'].nunique())

# filter trajectories by their particles' appearance
# plt.figure()
tp.mass_size(t1.groupby('particle').mean()); # plots size v. mass

# Particles with low mass, that are large or non-circular are probably out of focus
# or aggregated and can be filtered out.
t2 = t1[((t1['mass'] > 50) & (t1['size'] < 2.6) & (t1['ecc'] < 0.3))]
# plt.figure()
# tp.annotate(t2[t2['frame'] == 0], frames[0])

# trace the trajectories with plot_traj()
# plt.figure()
# tp.plot_traj(t2)

# drift motion can be subtracted out -> do we want to subtract it out?
d = tp.compute_drift(t2)
d.plot()
plt.show()

tm = tp.subtract_drift(t2.copy(), d)
ax = tp.plot_traj(tm)
plt.show()

# STEP 4: Analyze trajectories
