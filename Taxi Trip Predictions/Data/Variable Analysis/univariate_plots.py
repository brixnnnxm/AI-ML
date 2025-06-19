# Imports
import seaborn as sb
import matplotlib.pyplot as plt
import pandas as pd

# Access Data
df = pd.read_excel('/Users/briannamurphy/Documents/WMU/Classes/Machine Learning/Group Project/Data/sample_data.xlsx')
print('Data read.')

# Create Count Plot for Ride Count by Month
print('Making Count Plot for Hour of the Day')
sb.countplot(x = 'hour', data = df)
plt.title('Ride Count by Hour of the Day')
plt.xlabel('Hour')
plt.ylabel('Count')
plt.show()
print('Done!')

### Create Distribution Plot for Duration
##print('Making Distribution Chart')
##sb.histplot(df['duration'], bins = 50, kde = True)
##plt.title('Distribution of Duration')
##plt.xlabel('Duration')
##plt.ylabel('Frequency')
##plt.show()
