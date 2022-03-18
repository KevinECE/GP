from pmlb import fetch_data
from sklearn.model_selection import train_test_split


X, y = fetch_data('519_vinnie', return_X_y=True)
copy, test_X, train_y, test_y = train_test_split(X, y)

train_X = []
for c in copy:
    train_X.append(c[1])
print(train_X)

