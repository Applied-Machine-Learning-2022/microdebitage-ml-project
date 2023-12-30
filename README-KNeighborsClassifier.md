Download the necessary libraries
```
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split
```

Then get the data 
```
exp_1 = pd.read_excel("EXP-00001-Master.xlsx")
exp_2 = pd.read_excel('EXP-00002-Master.xlsx')
exp_3 = pd.read_excel('EXP-00003-Master.xlsx')
exp_4 = pd.read_excel('EXP-00004-Master.xlsx')
exp_5 = pd.read_excel('EXP-00005-Master.xlsx')
```
and drop the column that makes our data error

```
exp_1.drop(index=0, inplace=True)
exp_2.drop(index=0, inplace=True)
exp_3.drop(index=0, inplace=True)
exp_4.drop(index = 0, inplace = True)
exp_5.drop(index = 0, inplace= True)

exp_1.reset_index(drop=True, inplace=True)
exp_2.reset_index(drop=True, inplace=True)
exp_3.reset_index(drop=True, inplace=True)
exp_4.reset_index(drop=True, inplace=True)
exp_5.reset_index(drop=True, inplace=True)
```

We then looked for the columns that provided no information about the classification of the microdebitage and we removed those as well as adding what stage the production stage we predict the microdebitage was at. Stages 0-2 are for the chert tools and 4-5 are for the obsidian tool.

```
not_included = ['Id', 'Filter0','Filter1', 'Filter2','Filter3', 'Filter4', 'Filter5', 'Filter6', 'hash', 'Img Id', 'Curvature', 'Transparency', 'Angularity']
filtered = [x for x in exp_1.columns if x not in not_included]



exp_1_filtered = exp_1[filtered]
exp_2_filtered = exp_2[filtered]
exp_3_filtered = exp_3[filtered]
exp_4_filtered = exp_4[filtered]
exp_5_filtered = exp_5[filtered]

exp_1_filtered['Production Stage'] = 0
exp_2_filtered['Production Stage'] = 1
exp_3_filtered['Production Stage'] = 2
exp_4_filtered['Production Stage'] = 3
exp_5_filtered['Production Stage'] = 4
```
Then merge the dataframes into one

```
r1, c1 = exp_1_filtered.shape
r2, c2 = exp_2_filtered.shape
r3, c3  = exp_3_filtered.shape
data = exp_1_filtered.merge(exp_2_filtered, how= 'outer')
data
data = data.merge(exp_3_filtered, how = 'outer')
data = data.merge(exp_4_filtered, how = 'outer')
data = data.merge(exp_5_filtered, how='outer')
```

And split the data into train and test, which 80% of the data went into training and 20% went into testing.
```
 X_train, X_test, y_train, y_test = train_test_split(
    data[filtered],
    data['Production Stage'],
    test_size=0.2,
    stratify= data['Production Stage'],
    random_state=44)
```


Then we created the model with the best paremeters that I could find
```
model = KNeighborsClassifier(n_neighbors=40, weights= 'distance', algorithm=  'brute', leaf_size = 35)
model.fit(X_train, y_train)
labels = data['Production Stage']

random = data.sample(frac = 1)

one_hundred = random[:1000]

predictions = model.predict(X_test)
```

Then we created the model 
```
model = KNeighborsClassifier(n_neighbors=40, weights= 'distance', algorithm=  'brute', leaf_size = 35)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```
And found the accuracy score to be 34%
```
print('Accuracy score:',accuracy_score(y_test, predictions))
```
