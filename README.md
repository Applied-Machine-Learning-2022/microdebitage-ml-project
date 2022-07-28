[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-c66648af7eb3fe8bc4f294546bfd86ef473780cde1dea487d3c4ff354943c9ae.svg)](https://classroom.github.com/online_ide?assignment_repo_id=8127865&assignment_repo_type=AssignmentRepo)
<!--
Name of your teams' final project
-->
# final-project
## [National Action Council for Minorities in Engineering(NACME)](https://www.nacme.org) Google Applied Machine Learning Intensive (AMLI) at the University Of Kentucky

<!--
List all of the members who developed the project and
link to each members respective GitHub profile
-->
Developed by: 
- [Kimi Medina-Castellano](https://github.com/kimimedina) - `University of Kentucky`
- [Rodrigo Aguilar Barrios](https://github.com/Rodrigox30) - `University of California, Berkeley` 
- [Luke Taylor](https://github.com/LukeTaylor1) - `University of Kentucky` 
- [Jose Cruz](https://github.com/Resoj) - `University of Kentucky`

## Objective
To determine the production stages of ancient stone tools using automated measurements taken of experimental microdebitage.

## Goals
To create a model that will accurately determine the production stage that an ancient stone tool was undergoing based off the features of the microdebitage left behind.

## The Dataset
We received data for two different stone tools in the form of Excel files via Box (file sharing platform). The Excel files contain features about the physical properties of the microdebitage and each file contains data for each stage of the tool. In preparing the data, we removed features that were the exact same value across all the datasets – values that would make no difference – and removed features that immediately seemed irrelevant to our objective.

## Description
<!--
Problem: Archeologists 
-->

## Team Workflow
Each group member tried a different approach to develop an effective model. Below are links to each member's READ.ME file that details and explains their linear path for this project. 
* Kimi
* Rodrigo
* Luke
* Jose

## Usage instructions
<!--
Give details on how to install fork and install your project. You can get all of the python dependencies for your project by typing `pip3 freeze requirements.txt` on the system that runs your project. Add the generated `requirements.txt` to this repo.
-->
Due to a problem in the data collection of the microdebitage, we did not have an optimal model that would work to fit this problem. We will explain how this model will work once we get the optimal data that we think is needed to obtain better accuracy. After the tensorflow model, we will provide what we think was the best way to go about solving this problem by providing two different seperate problems solved by other models.
### **Tensorflow model**
We will first download the necessary libraries
```
import pandas as pd
import tensorflow as tf
```
We will convert the data into dataframes. Note that **EXP-00001, EXP-00002 and EXP-00003** refers to the data of the microdebitage at the production stage of the chert tool  while **EXP-00004 and EXP-00005** are the stages of the obsidian tool.
```
exp_1 = pd.read_excel("EXP-00001-Master.xlsx")
exp_2 = pd.read_excel('EXP-00002-Master.xlsx')
exp_3 = pd.read_excel('EXP-00003-Master.xlsx')
exp_4 = pd.read_excel('EXP-00004-Master.xlsx')
exp_5 = pd.read_excel('EXP-00005-Master.xlsx')
```
Due to an error caused by Excel, we will delete the second row with the following code
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
Since we found some data to not have any impact on our model accuracy, we decided to take it out of our dataframes. We included Curvature, Transparency, Angularity in the columns to be removed because it was missing in the chert data. We decided it was better to remove this because we did not want to make up data. We also added the production stage such that stage 0 was the first stage of chert, stage 1 was for the second stage of chert, stage 2 was for the third stage of chert, stage 3 is the first stage of the obsidian tool and stage 4 was the second stage of the obsidian tool.
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

Then we merged all dataframes into one
```
data = exp_1_filtered.merge(exp_2_filtered, how= 'outer')
data = data.merge(exp_3_filtered, how = 'outer')
data = data.merge(exp_4_filtered, how = 'outer')
data = data.merge(exp_5_filtered, how='outer')
```

We then converted all values in the dataframes to numeric types
```
for x in data.columns: 
    data[x] = pd.to_numeric(data[x])
```
In order to use the Tensorflow, we will need to add 5 additional columns that each will 1 if it's the stage that the column is about or 0 for everything else
```
stage_0 = [1 if x ==0 else 0 for x in data['Production Stage'] ]
stage_1 = [1 if x ==1 else 0 for x in data['Production Stage']]
stage_2 = [1 if x ==2 else 0 for x in data['Production Stage']]
stage_3 = [1 if x ==3 else 0 for x in data['Production Stage']]
stage_4 = [1 if x ==4 else 0 for x in data['Production Stage']]

the_stages = ['stage_0', 'stage_1', 'stage_2', 'stage_3', 'stage_4']

data['stage_0'] = stage_0
data['stage_1'] = stage_1
data['stage_2'] = stage_2
data['stage_3'] = stage_3
data['stage_4'] = stage_4
```

We then made the neural network. We found this method to work well enough. We had 5 outputs because for the model to classify. We shuffled the data such that eighty percent of it is training as training and the other twenty as testing.
```
model = tf.keras.Sequential([
    tf.keras.layers.Dense(len(filtered) * 2, input_shape=(len(filtered),)),
    tf.keras.layers.Dense(16, activation = 'relu'),
    tf.keras.layers.Dense(8, activation = 'relu'),
    tf.keras.layers.Dense(5, activation = tf.nn.softmax)])

model.compile(
    loss='categorical_crossentropy',
    optimizer='Adam',
    metrics=['accuracy'],
)

rows = data.shape[0]
eighty = int(rows * .8)

random = data.sample(frac=1)
train = random[:eighty]
test = random[eighty:]
```
Then, we fitted the model using our data and let it run.
```
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
history = model.fit(
    train[filtered],
    train[the_stages],
    epochs=100,
    callbacks=[callback],
    validation_data = [test[filtered], test[the_stages]]
)
```
And found that our highest accuracy was around 38%.
```
import matplotlib.pyplot as plt
history.history['accuracy']

plt.plot(history.epoch, history.history['accuracy'])
```
![Model](output.png)