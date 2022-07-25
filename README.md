[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-c66648af7eb3fe8bc4f294546bfd86ef473780cde1dea487d3c4ff354943c9ae.svg)](https://classroom.github.com/online_ide?assignment_repo_id=8127865&assignment_repo_type=AssignmentRepo)
<!--
Name of your teams' final project
-->
# final-project
## [National Action Council for Minorities in Engineering(NACME)](https://www.nacme.org) Google Applied Machine Learning Intensive (AMLI) at the `PARTICIPATING_UNIVERSITY`

<!--
List all of the members who developed the project and
link to each members respective GitHub profile
-->
Developed by: 
- [Kimi Medina-Castellano](https://github.com/kimimedina) - `University of Kentucky`

### Description
Our task for this project was to create a model to accurately classify samples of microdebitage simulants into 5 tool production stages using numerical data provided by Dr. Phyllis Johnson. The level of desired accuracy was not defined, but we managed to reach 80% accuracy using Rodrigo's K-Nearest Neighbors (KNN) model. I tried two models: decision trees and random forest. 

## Exploratory Data Analysis
In this section, I:
1. **loaded data**: loaded five excel files of measurements of microdebitage samples that represented different stages in the knapping process into separate dataframes.
2. **checked for missing data:** there was no missing data in any of the dataframes.
3. **assigned an integer for each tool stage:** this is the column we will predict on 
```
# add target column, stage, to each dataframe
chert_hh_1['Stage'] = 1
chert_hh_2['Stage'] = 2
chert_sh['Stage'] = 3
obsidian_cr['Stage'] = 4
obsidian_sh['Stage'] = 5
```
4. **merged data for tool stages into one large dataframe** 
5. **cleaned the merged dataframe:** 
``` 
# remove unecessary columns + row 0
# row 0 included labels for units of measurement (ex: mm), which were all millimeters
tools_df.drop(columns = ['Id','Filter0','Filter1', 'Filter2', 'Filter3', 'Filter4', 'Filter5', 'Filter6', 'hash'], inplace = True)
tools_df.drop(columns = ['Transparency', 'Curvature', 'Angularity', 'Img Id'], inplace = True)
tools_df.drop(0, inplace = True) 
```
6. **filled missing data** 
7. **visualized correlation between columns:** 
![corr_tools_df](https://user-images.githubusercontent.com/106893508/180860119-73a6428f-c14a-4304-ade8-3913ac26a98a.png)

### Complications
There were three extra columns that our obsidian data had whereas our chert data did not. These were Transparency, Curvature, and Angularity. Initially, I kept these columns for obsidian, added empty columns with the same names to the chert data sets, and filled the empty rows with zeros. Running my two models with this cleaned data gave me 80.1% and 58.7% accuracy in my decision trees and random forest model, respectively. After I tuned the hyperparameters for the decision tree model, my accuracy increased to 80.8%.

We double-checked with Dr. Johnson on whether or not chert microdebitage can be transparent and why there was no data for those three columns. She said they did not have those parameters turned on and should have had that data. This prompted the team to remove the three columns altogether. I ran the updated dataset into both of my models and got about 37% for a basic and tuned model. With my low accuracy, we went with Rodrigo's KNN model.

## Modeling
### Decision Trees
Decision trees are supervised machine learning method in which data is split according to certain parameters defined by the user. I used the DecisionTreeClassifier class from sklearn library and had multiple iterations of the trees throughout this process.

#### With Transparency, Curvature, and Angularity Columns
This decision tree model was done before I removed the three columns. I first began with a basic model with no tuned parameters that had an accuracy of 80.1%. I initially tuned the model by iterating through each integer value a parameter can have within a specific range. For example, here is code of me doing so with the 'min_samples_split' parameter. The result is a graph of the model accuracy vs each integer value of the parameter. 
```
# determine best max_depth
split_list = []
acc_list2 = []
for i in range(2, 40):
  dt = tree.DecisionTreeClassifier(
    criterion = 'entropy', 
    max_depth = 11,
    min_samples_split = i,
    random_state = 10)
  # fit tree
  dt.fit(x_train, y_train)
  # get predictions
  y_pred = dt.predict(x_test)
  # get accuracy for each iteration and append to list
  acc = metrics.accuracy_score(y_test, y_pred)
  acc_list2.append(acc)
  split_list.append(i)

# show graph of 
plt.plot(split_list, acc_list2)
plt.show()

print(max(acc_list2))
for xy in zip(split_list, acc_list2):
   print(xy)
```
![min_samples_split vs accuracy](https://user-images.githubusercontent.com/106893508/180866102-b9406195-034d-41cb-8e89-915d759f3dfd.png)

##### Some Shortcomings...
only using accuracy as a performance metric
not tuning random forest even though it was low accuracy
not waiting for gridsearchcv to run (took >2hrs)

