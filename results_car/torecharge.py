
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Bring some raw data.
frequencies = [99.87,94.52,  93.49]
# In my original code I create a series and run on that, 
# so for consistency I create a series from the list.
freq_series = pd.Series.from_array(frequencies)

x_labels = [ 'Cost','Time', 'Eco']
# Plot the figure.
plt.figure(figsize=(12, 8))
ax = freq_series.plot(kind='bar')
#ax.set_title('Total Price spent to recharge')
#ax.set_ylabel('€')
ax.set_xticklabels(x_labels)

rects = ax.patches

# Make some labels.
labels = ['99.87kWh' ,'94.52kWh','93.49kWh']
# In my original code I create a s']

for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2., 1.01*height,
                '%s' % label,
                ha='center', va='bottom')

plt.show()