# plot_map.py
"""Simple test map plotter"""

import shapefile
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

shapes = shapefile.Reader('ne_110m_land/ne_110m_land').shapes()
f = plt.figure(facecolor='#ccccff')
ax = f.add_axes((0,0,1,1), axisbg='none', frameon=False)
for shape in shapes:
    a = np.array(shape.points)
    ax.fill(a[:,0], a[:,1], fc='white', ec='none')
plt.show()