# Imports
import  pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

# Data Path
print('Reading Path.')
df = pd.read_excel('/Users/briannamurphy/Documents/WMU/Classes/Machine Learning/Group Project/Data/train_data.xlsx', sheet_name = 'Log Transformed')
data = df[['passenger_count', 'trip_distance', 'fare_amount', 'tip_amount', 'tolls_amount', 'total_amount', 'pickup_location_id', 'dropoff_location_id', 'hour', 'day', 'month', 'duration', 'log_duration']]
print('Data Found.')

# Linear (Pearson) Correlation
print('Starting Pearson Correlation Matrix.')
lin_data = data.corr(method = 'pearson')
plt.figure(figsize = (10, 8))
sb.heatmap(lin_data, annot = True, cmap = 'coolwarm', fmt = ".2f")
plt.title('Pearson Correlation Matrix')
print('Pearson Correlation Matrix Complete.')
plt.show()

# Nonlinear (Spearman) Correlation
print('Starting Spearman Correlation Matrix.')
nl1_data = data.corr(method = 'spearman')
plt.figure(figsize = (10, 8))
sb.heatmap(nl1_data, annot = True, cmap = 'coolwarm', fmt = ".2f")
plt.title('Spearman Correlation Matrix')
print('Spearman Correlation Matrix Complete.')
plt.show()

print('Execution Complete.')
