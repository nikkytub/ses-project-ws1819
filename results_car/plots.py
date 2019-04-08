
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Bring some raw data.
frequencies = [8.968 ,7.896,7.070]
# In my original code I create a series and run on that, 
# so for consistency I create a series from the list.
freq_series = pd.Series.from_array(frequencies)

x_labels = [ 'Time', 'Cost','Eco']
# Plot the figure.
plt.figure(figsize=(12, 8))
ax = freq_series.plot(kind='bar', color='blue', alpha=0.4)
#ax.set_title('Total Normalized Cost Factor')
#ax.set_ylabel('Factor')
ax.set_xticklabels(x_labels, fontsize='22')
ax.get_yaxis().set_visible(False)

rects = ax.patches

# Make some labels.
labels = ['8.968' ,'7.896','7.070']
############
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2., 0.8*height,
                '%s' % label,
                ha='center', va='bottom', fontsize='24')

plt.show()