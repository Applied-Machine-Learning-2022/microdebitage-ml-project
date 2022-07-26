[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-c66648af7eb3fe8bc4f294546bfd86ef473780cde1dea487d3c4ff354943c9ae.svg)](https://classroom.github.com/online_ide?assignment_repo_id=8127865&assignment_repo_type=AssignmentRepo)
<!--
Name of your teams' final project
-->
# final-project-the-rock-group-uk
## [National Action Council for Minorities in Engineering(NACME)](https://www.nacme.org) Google Applied Machine Learning Intensive (AMLI) at the `University of Kentucky`

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
![heatmap_new](https://user-images.githubusercontent.com/106893508/181030676-b96199ab-8c05-41b4-a145-a3aa7183b2c2.png)


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

I also did this method with the parameters 'max_depth', 'min_samples_leaf', and 'min_weight_fraction_leaf'. I inserted these parameters into a decision tree and got an accuracy of 80.8%. I attempted to use GridSearchCV for decision trees, and it gave a lower accuracy, around 80.7% (although it took a long time to do so). I did not run the random forest model for this scenario because decision trees were more promising.

#### Column Deletion
With the deleted columns, a non-tuned decision tree model had an accuracy of 28.47%. A tuned tree had 36.9% accuracy.

### Random Forest 
I decided to test out random forest because it relies on multiple randomly generated decision tree which should, in theory, increase model accuracy and reduce overfitting.

#### With Transparency, Curvature, and Angularity Columns
My accuracy for a basic model was 81.98% and a tuned model via GridSearchCV (it worked!) was 81.87%. 

#### Column Deletion
A basic random forest model without parameter tuning was 37.4% accurate. I was able to use GridSearchCV and get some parameters. The result was a similar accuracy of 37.39%. This didn't feel right, but since we had a better model given our time constraints, I did not explore further. 

##### Some Shortcomings...
* only using accuracy as a performance metric instead of precision or F1 score
* not double-checking random forest tuning even though it was low accuracy
* not waiting for GridSearchCV to run on decision tree model (took >2hrs)

