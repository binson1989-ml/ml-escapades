import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

test_file = 'train.csv'
data = pd.read_csv(test_file)
y = data.SalePrice
X = data[['LotArea','YearBuilt','1stFlrSF','2ndFlrSF','FullBath','BedroomAbvGr','TotRmsAbvGrd']]
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

model = RandomForestRegressor()
model.fit(train_X, train_y)
val_predictions = model.predict(val_X)
print('Mean absolute error: %d' %(mean_absolute_error(val_y, val_predictions)))
