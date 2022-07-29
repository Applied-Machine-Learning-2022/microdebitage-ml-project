### **KNeighbors Classifier**
We originally tried making a single model to classify both chert and obsidian however, the accuracy reading were very low. So our team decided to make two models, one for chert and one for obsidian because the transparency feature was too important to be left out. This section goes over the KNeighbors model. Most of the earlier stages of importing and shaping the data will be skipped because it was mentioned above.

The K neighbors classifier is used for supervised learning, which is what we were given since we know the stages of the tools. The classifier works by classifing nodes based on their distances to the 'neighbors' (similar data points) around them. [Link](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm) to wikipedia article for more in depth explaination.

First we import the data from the spread sheet. Three of the datasets are chert and two of them are obsidan and then drop the first row since it just contained the units for all of the data points and weren't numerical values. 

```
# Chert 
exp_1 = pd.read_excel("EXP-00001-Master.xlsx")
exp_2 = pd.read_excel('EXP-00002-Master.xlsx')
exp_3 = pd.read_excel('EXP-00003-Master.xlsx')

# Obsidian 
exp_4 = pd.read_excel('EXP-00004-Master.xlsx')
exp_5 = pd.read_excel('EXP-00005-Master.xlsx')

# dropping first row
exp_1.drop(index=0, inplace=True)
exp_2.drop(index=0, inplace=True)
exp_3.drop(index=0, inplace=True)
exp_4.drop(index = 0, inplace = True)
exp_5.drop(index = 0, inplace= True)

# resetting the index values
exp_1.reset_index(drop=True, inplace=True)
exp_2.reset_index(drop=True, inplace=True)
exp_3.reset_index(drop=True, inplace=True)
exp_4.reset_index(drop=True, inplace=True)
exp_5.reset_index(drop=True, inplace=True)
```

Because we are making a model for both, we made two separate filters to get the respective columns in each dataframe. The filters were pretty much the same except Curvature, Angularity and Transparency were filtered out of the Chert model. After, the filtered datasets were saved into variables

```
# Obsidian 
O_not_included = ['Id', 'Filter0','Filter1', 'Filter2','Filter3', 'Filter4', 'Filter5', 'Filter6', 'hash', 'Img Id']
O_filtered = [x for x in exp_4.columns if x not in O_not_included]

# Chert
C_not_included = ['Id', 'Filter0','Filter1', 'Filter2','Filter3', 'Filter4', 'Filter5', 'Filter6', 'hash', 'Img Id', 'Curvature', 'Transparency', 'Angularity']
C_filtered = [x for x in exp_1.columns if x not in C_not_included]

# Chert Filtered 
exp_1_filtered = exp_1[C_filtered]
exp_2_filtered = exp_2[C_filtered]
exp_3_filtered = exp_3[C_filtered]

# Obsidian filtered
exp_4_filtered = exp_4[O_filtered]
exp_5_filtered = exp_5[O_filtered]
```

Here, we are adding the production stages. Similar to what was done for the Tensor flow model. There are 2 tools, one tool had three distinct stages and the other had two distinct stages which means there are 5 distinct outcomes.
```
# Setting production stage values
# Chert
exp_1_filtered['Production Stage'] = 0
exp_2_filtered['Production Stage'] = 1
exp_3_filtered['Production Stage'] = 2

# Obsidian
exp_4_filtered['Production Stage'] = 3
exp_5_filtered['Production Stage'] = 4
```

Then we made the two data frames. exp_1, exp_2, exp_3 went into the chert data frame and exp_4 and exp_5 went into the obsidian.
```
# Merging Chert Data
C_data = exp_1_filtered.merge(exp_2_filtered, how= 'outer')
C_data = C_data.merge(exp_3_filtered, how = 'outer')

# Merging Obsidian Data
O_data = exp_4_filtered.merge(exp_5_filtered, how = 'outer')
```

In order to get the training and the testing data, we used the scikit learn's test train split, with the stratify on the production stage column. The split is 10% in test, 90% in train.
```
# Obsidian test train split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    O_data[O_filtered],
    O_data['Production Stage'],
    test_size=0.1,
    stratify= O_data['Production Stage'],
    random_state=44)
```

Since the data contains different scales of data, we had to use the StandardScaler to standardize the data.
```
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test= sc_X.transform(X_test)
```

Here is the model for obsidian. The parameters were optimized using a gridsearch that will be explained later. After we ran the model, we computed the accuracy score which was 72%. This was a significant increase when compared with the first K neighbors attempt.
```
# Obsidian Model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import scale
from sklearn.metrics import accuracy_score

O_model = KNeighborsClassifier(n_neighbors=5,
                            weights= 'distance', 
                            p = 1, 
                            leaf_size= 5, 
                            algorithm='auto')
O_model.fit(O_data[O_filtered], O_data["Production Stage"])
labels = O_data['Production Stage']

O_predictions = O_model.predict(X_test)

print('Accuracy score:',accuracy_score(y_test, O_predictions))

# Accuracy score: 0.7260606060606061
```

To find the best parameters for the model, we ran a grid search looking at the weight, algorithm, and leaf size. 
The weight takes into account the distance of the nodes, so uniform means all distances are weighted the same while distance increase the weight the close nodes are to one another.
The algorithm is how the nearest neighbor is calculated. 
links to [ball tree](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.BallTree.html#sklearn.neighbors.BallTree) and [kd tree](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KDTree.html#sklearn.neighbors.KDTree)
auto allows the model to decide which is best for what instance, and brute is the brute force method.
The leaf size is for the ball and kd tree algorithm
```
# Obsidian Grid search
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

params = [{ 
    'weights' : ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'leaf_size': [5,10,15,20,25,30,35,40],
            }]

search = GridSearchCV(O_model, 
                    param_grid = params,
                    scoring = 'accuracy',
                    cv = 5)

search.fit(X_train,y_train)
print(search.best_params_)

# OUTPUT: {'algorithm': 'auto', 'leaf_size': 5, 'weights': 'distance'}
```


The same process was repeated for the chert. As you can see below, the code is pretty much identical besides the changing of the variables. However, the results are drastically worse. The accuracy of the model is around 44% which is almost 30% worse than the obsidian one from above. The shows the importance of the transparancy, curvature and angularity columns
```
# Chirt test train split
from sklearn.model_selection import train_test_split

CX_train, CX_test, cy_train, cy_test = train_test_split(
    C_data[C_filtered],
    C_data['Production Stage'],
    test_size=0.1,
    stratify= C_data['Production Stage'],
    random_state=44)

sc_CX = StandardScaler()
C_train = sc_CX.fit_transform(CX_train)
CX_test= sc_CX.transform(CX_test)

# Chert Model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import scale
from sklearn.metrics import accuracy_score

C_model = KNeighborsClassifier(n_neighbors=5, 
                            weights= 'distance',
                            p = 1, leaf_size= 5, 
                            algorithm='auto')

C_model.fit(C_data[C_filtered], C_data["Production Stage"])
labels = C_data['Production Stage']

random = C_data.sample(frac = 1)

one_hundred = random[:10000]

C_predictions = C_model.predict(CX_test)

print('Accuracy score:',accuracy_score(cy_test, C_predictions))

# Accuracy score: 0.4437566324725858
```

### **Gradient Boosting**
This model was made before we decided to make two models, so the accuracy scores could be significantly better. This model is something I want to continue working with in the future, the first reason is to include the angularity, transparency, and curvature. The second is that I didn't have the most experience with the model, so I think another attempt at it would also increase the over all effectiveness of the model.

```
# Gradient Boosting
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score

GBModel = GradientBoostingClassifier(
                                    n_estimators=100, 
                                    learning_rate = .30 , 
                                    max_depth = 3, 
                                    max_features= 'auto',
                                    criterion= 'mse',
                                    loss = 'deviance'
                                    )


GBModel.fit(scale(data[filtered].values), data["Production Stage"])

random = data.sample(frac = 1)

ten_thousand = random[:10000]

predictionsGB = GBModel.predict(ten_thousand[filtered])

print('Accuracy score:',accuracy_score(ten_thousand['Production Stage'], predictionsGB))
```


With this model especially, I had a hard time running the grid search due to the fact that it would never run to completion.
max depth: The maximum depth of the individual regression estimators.
learning rate: shrinks contribution by tree
n estimators: boosting stages to perform
loss: this determines what loss fuction to optimize
criterion: the quality of the split
```
from sklearn.pipeline import Pipeline
gbc_pipe = Pipeline([('GBC', GradientBoostingClassifier())])

params = [{ 'max_depth': [ 1, 2, 3, 4, 5],
            'learning_rate': [.01, .10, .20, .30, .40, .50],
            'n_estimators': [50, 150, 250, 350, 450,],
            'loss': ['log_loss', 'deviance', 'exponential'],
            'criterion': ['friedman_mse', 'squared_error', 'mse']
            }]

search = GridSearchCV(GBModel, 
                    param_grid = params,
                    scoring = 'accuracy',
                    cv = 5)

search.fit(X_train,y_train)
print(search.best_params_)

print("score: {}".format(search.score(X_train, y_train)))

```