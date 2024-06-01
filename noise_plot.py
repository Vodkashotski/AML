import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Code for plotting unfiltered sensor measurements

def importData(set):
    Header = ["unit number","time, in cycles", "operational setting 1", "operational setting 2", "operational setting 3",
               "sensor measurement 1", "sensor measurement 2", "sensor measurement 3", "sensor measurement 4",
                 "sensor measurement 5","sensor measurement 6", "sensor measurement 7", "sensor measurement 8", "sensor measurement 9",
                "sensor measurement 10", "sensor measurement 11", "sensor measurement 12", "sensor measurement 13",
                "sensor measurement 14", "sensor measurement 15", "sensor measurement 16", "sensor measurement 17",
                "sensor measurement 18", "sensor measurement 19", "sensor measurement 20", "sensor measurement 21"]
    data = pd.read_csv("{}" .format(set), header=None, delim_whitespace=True)
    data.columns = Header
    return data

data = importData('Data/train_FD003.txt')

units_to_plot = [25, 50, 75]

data_filtered = data[data['unit number'].isin(units_to_plot)]

grouped_data_filtered = data_filtered.groupby('unit number')

plt.figure(figsize=(12,6))

for unit_number, group in grouped_data_filtered:
    plt.plot(group['time, in cycles'], group['sensor measurement 11'], label=f'Unit {unit_number}')

plt.xlabel('Time in cycles')
plt.ylabel('HPC outlet static pressure (psia)')
plt.title('Sensor Measurement 11')
plt.legend()
plt.show()

