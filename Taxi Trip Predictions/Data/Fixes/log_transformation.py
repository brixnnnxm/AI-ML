#  Imports
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from openpyxl import load_workbook

# Load workbooks to update duration column
print('Executions started, reading workbook.')
path = '/Users/briannamurphy/Documents/WMU/Classes/Machine Learning/Group Project/Data/sample_data.xlsx'
data = pd.read_excel(path, sheet_name = 'Sheet2')
print('File read, starting transformation.')

data_log = data.copy()
data_log['log_duration'] = np.log1p(data_log['duration'])
print('Log transformation on sample dataset complete.')

# Save
with pd.ExcelWriter(path, engine = "openpyxl", mode = "a") as writer:
    data_log.to_excel(writer, sheet_name = "Log Transformed", index = False)
    print('sample dataset saved.')

# Load workbooks to update duration column
print('Executions started, reading workbooks.')
new_data = pd.read_excel(path,  sheet_name = 'Log Transformed')
print('File read, starting distribution graphs.')

# Show distribution
sb.histplot(new_data['duration'], bins = 50, kde = True)
plt.title('Distribution of Duration (not transformed)')
plt.xlabel('Duration')
plt.ylabel('Frequency')
plt.show()
print('train duration distribution.')

sb.histplot(new_data['log_duration'], bins = 50, kde = True)
plt.title('Distribution of Duration (not transformed)')
plt.xlabel('Duration')
plt.ylabel('Frequency')
plt.show()
print('train duration distribution.')
