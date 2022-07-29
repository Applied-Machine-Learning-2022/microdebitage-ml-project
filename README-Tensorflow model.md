
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