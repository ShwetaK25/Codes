import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mpl_toolkits
import os

CUR_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)))
input_csv = os.path.join(CUR_DIR, 'washington1.csv')
data = pd.read_csv(input_csv)
print(data.head())

data['bedrooms'].value_counts().plot(kind='bar')
plt.title('number of Bedroom')
plt.xlabel('Bedrooms')
plt.ylabel('Count')
sns.despine
plt.show()

plt.scatter(data.bedrooms,data.price)
plt.title("Bedroom & price")
plt.xlabel("Bedrooms")
plt.ylabel("Price")
plt.show()
sns.despine

plt.scatter(data.floors,data.price)
plt.title("Floors & price")
plt.xlabel("Floors")
plt.ylabel("Price")
plt.show()
sns.despine


plt.scatter(data.yr_built,data.price)
plt.title("yr_built & price")
plt.xlabel("yr_built")
plt.ylabel("Price")
plt.show()
sns.despine



plt.scatter((data['sqft_living']+data['sqft_basement']), data['price'])
plt.xlabel('sqft_living + sqft_basement')
plt.ylabel('price')
plt.show()
sns.despine

plt.scatter(data.waterfront, data.price)
plt.title("waterfront vs price(0=no waterfront)")
plt.xlabel('waterfront')
plt.ylabel('price')
plt.show()
sns.despine

plt.scatter(data.condition, data.price)
plt.xlabel('condition')
plt.ylabel('price')
plt.show()
sns.despine

plt.scatter(data.statezip, data.price)
plt.xlabel('statezip')
plt.ylabel('price')
plt.show()
sns.despine

