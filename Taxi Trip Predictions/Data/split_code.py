import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

filePath = "/Users/briannamurphy/Documents/WMU/Classes/Machine Learning/Group Project/taxi_trip_data.xlsx"
df = pd.read_excel(filePath)

# Split
train_data, temp_data = train_test_split(df, test_size = 0.2, random_state = 42)
test_data, val_data =  train_test_split(temp_data, test_size = 0.5, random_state = 42)

# Save
train_data.to_excel('train_data.xlsx', index = False)
test_data.to_excel('test_data.xlsx', index = False)
val_data.to_excel('val_data.xlsx', index = False)
print("Data has been split, and saved.")
