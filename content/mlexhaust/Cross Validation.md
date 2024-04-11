---
title: Cross Validation
draft: false
tags:
  - machinelearning
---

### What is cross validation ?

**Cross validation** is a method of splitting our data in some fashion so that our machine learning model learns the pattern from data in a good manner (_not exactly because then it will leads to **overfitting**). 

Firstly, let's look, what is **overfitting** through code - 

`we are trying to split our dataset in two part and train different decision tree model and see the training and testing accuracies with their plot and understand what overfitting is `

```python
# import libraries
from sklearn import tree
from sklearn import metrics
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# read the data -> you can fetch red wine dataset from kaggle

df = pd.read_csv('winequality-red.csv')
# we'll map the quality with some numbers
quality_mapping = {3:0, 4:1, 5:2, 6:3, 7:4, 8:5}
df.loc[:,"quality"] = df.quality.map(quality_mapping)

# Now let's shuffle and split the data
df = df.sample(frac=1).reset_index(drop=True)

#top 1000 rows are selected
df_train = df.head(1000)
# bottom 599 rows are selected
df_test = df.tail(599)

# list to store accuracies

train_accuracies = [0.5]
test_accuracies = [0.5]

# cols/features for training 
cols = ["fixed_acidity","volatile_acidity","citric_acid","residual_sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"]

# iterate over a few depth values
for depth in range(1,25):
	# init the model
	clf = tree.DecisionTreeClassifier(max_depth=depth)

	#fit the model
	clf.fit(df_train[cols],df_train.quality)
	
	#create training and test predictions
	train_predictions = clf.predict(df_train[cols])
	test_predictions = clf.predict(df_test[cols])

	# calculate training and test accuracies
	train_accuracy = metrics.accuracy_score(df_train.quality,train_predictions)
	test_accuracy = metrics.accuracy_score(df_test.quality,test_predictions)
	# Append accuracies
	train_accuracies.append(train_accuracy)
	test_accuracies.append(test_accuracy)

# create plots 
plt.figure(figsize=(10,5))
sns.set_style("whitegrid")
plt.plot(train_accuracies,label="train accuracy")
plt.plot(test_accuracies,label="test accuracy")
plt.legend(loc="upper left",prop={'size':15})
plt.xticks(range(0,26,5))
plt.xlabel("max_depth",size=20)
plt.ylabel("accuracy",size=20)
plt.show()
```

So after plotting, you can see like below :

![[quartz/static/images/train_test_accuracy.png]]

Here, as we are increasing our **max_depth** parameter to try our model learns more and more accurately, but it doesn't work like that. You can see our training accuracy is increasing as increasing **max_depth** value but testing increasing doesn't give a shit about it. This is what **overfitting** is. So don't be happy by looking good accuracy on training data because maybe it can't perform well on real world cases/test data. 

**Occam's razor** in simple words states that one should not try to complicate things that can be solved in a much simpler manner. In other words, the simplest solutions are the most generalizable solutions.

![[quartz/static/images/occam_razor.png]]

Let's move ahead to **cross validation** now,

So we'll have many cross validation techniques and each technique is independent and vary from data to data.
# First Approach (Hold Out Set Validation)

As above when we are splitting dataset, we divided our data in two parts i.e. for training and testing. This approach is called **hold-out set**. Mainly we use this kind of cross validation when we have large amount of data and model inference is a time consuming process.

![[quartz/static/images/cv.png]]

# Second Approach (K Fold Cross Validation)

In this technique, we split our dataset in many parts (let's say k = 5). So we divide our dataset in 5 equal portions then train our required model on k-1 folds and test on remaining last fold. Now we again shuffle the folds so that our validation fold changed again as well as testing fold, again we'll test. That's how we can build good model.

![[quartz/static/images/kfold.png]]

We can do **K-Fold cross validation** using sklearn library :

```python
import pandas as pd
from sklearn import model_selection

df = pd.read_csv('train.csv')

# create one column called kfold which stores which row is from which number of fold

df['kfold'] = -1

# shuffle the data
df = df.sample(frac=1).reset_index(drop=True)

# inititate the kfold class 
kf = model_selection.KFold(n_splits=5) # number of folds

# fill the new kfold column
for fold, (trn_, val_) in enumerate(kf.split(X=df)):
	df.loc[val_,'kfold'] = fold

# Save the new csv
df.to_csv("train_folds.csv",index=False)
```

We can use this technique with almost all dataset where we don't require much equal proportion of target values in each fold. For that we have next approach.

# Third Approach (Stratified K Fold Cross Validation)

Now, whenever we have a dataset with some target labels means we are going to deal with classification problem in which all targets have different distribution over the data like below e.g.

![[quartz/static/images/distribution.png]]
If we perform normal _k fold validation_ some samples only contains one class of target and would create biased model towards some class, so **stratified k fold cross validation** identifies the distribution % of classes in target and maintain that ratio in each folds also.

```python
import pandas as pd
from sklearn import model_selection

df = pd.read_csv('train.csv')

# create one column called kfold which stores which row is from which number of fold

df['kfold'] = -1

# shuffle the data
df = df.sample(frac=1).reset_index(drop=True)

# inititate the kfold class 
kf = model_selection.StratifiedKFold(n_splits=5) # number of folds

# fill the new kfold column
for fold, (trn_, val_) in enumerate(kf.split(X=df)):
	df.loc[val_,'kfold'] = fold

# Save the new csv
df.to_csv("train_folds.csv",index=False)
```

# Fourth Approach (Hold out based cross validation)

Here, we'll use this when we have very huge data so for that we can use _k fold_ but it would be time consuming/resource eating for multiple validation and training folds. So we can create folds only once and put the one fold for hold out validation. Mainly it is used in **time series data** where we are keeping some last weeks/years of data as a hold out and train for previous of them.

![[quartz/static/images/holdout.png]]

**Now, how we can apply validation to regression based dataset** - Yes by the way, we can apply _k fold cv_ easily for the data and even we can use _stratified k fold cv_ by firstly binning the continuous target feature in some buckets. 

**Note -** Cross validation is very crucial step when building machine learning model. It is the initial step before when we starting our feature engineering process. To decide right technique so that we can test our model on good data means having ample amount of each cases that it can cover.