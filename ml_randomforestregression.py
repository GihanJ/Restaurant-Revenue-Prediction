
from sklearn.preprocessing import LabelEncoder  
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

data_dir = "input/"
trainData = pd.read_csv(data_dir + 'train.csv', parse_dates = [1])
testData = pd.read_csv(data_dir + 'test.csv', parse_dates = [1])

trainData.head()

testData.head()

trainData.isnull().sum()

testData.isnull().sum()

trainData.shape

testData.shape

trainData.describe()

testData.describe()

print (testData['City'].unique()) 
print ('\n')
print (trainData['City'].unique())

print (testData["City"].unique().shape)
print (trainData["City"].unique().shape)

test_r = testData.copy(deep=True)
train_r = trainData.copy(deep=True)
plt.figure(figsize=(24,20))
train_r["City"].value_counts().plot(kind='bar')
plt.show()
plt.figure(figsize=(24,20))
test_r["City"].value_counts().plot(kind='bar')
plt.show()

grp = trainData.groupby(['City'])['revenue'].sum()
df_grp = grp.to_frame().reset_index()
df_grp.index.name = 'index'

sns.set(style="whitegrid")
plt.figure(figsize=(24, 20))
plt.title('Average Revenue by City', loc = 'center')
ax = sns.barplot(y="City", x="revenue", data=df_grp)
plt.show()

test_r = testData.copy(deep=True)
train_r = trainData.copy(deep=True)
print ("TEST DATA")
print (test_r["Type"].unique().shape)
print(test_r["Type"].unique())
test_r["Type"].value_counts().plot(kind='bar')
plt.show()
print (test_r["Type"].value_counts())
print ("\n")
print ("TRAIN DATA")
print (train_r["Type"].unique().shape)
print(train_r["Type"].unique())
train_r["Type"].value_counts().plot(kind='bar')
plt.show()
print (train_r["Type"].value_counts())

numerical_features = trainData.select_dtypes([np.number]).columns.tolist()
categorical_features = trainData.select_dtypes(exclude = [np.number,np.datetime64]).columns.tolist()

categorical_features

numerical_features

sns.set(style='whitegrid', palette="deep", font_scale=1.1, rc={"figure.figsize": [8, 5]})
sns.distplot(
    trainData['revenue'], norm_hist=False, kde=True
).set(xlabel='revenue', ylabel='P(revenue)')

trainData[trainData['revenue'] > 10000000 ]

fig, ax = plt.subplots(3, 1, figsize=(40, 30))
for variable, subplot in zip(categorical_features, ax.flatten()):
    df_2 = trainData[[variable,'revenue']].groupby(variable).revenue.sum().reset_index()
    df_2.columns = [variable,'total_revenue']
    sns.barplot(x=variable, y='total_revenue', data=df_2 , ax=subplot)
    for label in subplot.get_xticklabels():
        label.set_rotation(90)

temp = trainData.copy()
cts = ['İstanbul','İzmir','Ankara','Bursa','Samsun']
fig, ax = plt.subplots(5, 1, figsize=(25, 45))
for variable, subplot in zip(cts, ax.flatten()):
    x = temp[trainData["City"]==variable]
    x = x.sort_values(by=['Open Date'])
    g = sns.lineplot(x="Open Date", y="revenue", style = "Type",label=variable, data=x, ax=subplot)
    g.title.set_text(variable)
    for label in subplot.get_xticklabels():
      label.set_rotation(90)

k = len(trainData[numerical_features].columns)
n = 3
m = (k - 1) // n + 1
fig, axes = plt.subplots(m, n, figsize=(n * 5, m * 3))
for i, (name, col) in enumerate(trainData[numerical_features].iteritems()):
    r, c = i // n, i % n
    ax = axes[r, c]
    col.hist(ax=ax)
    ax2 = col.plot.kde(ax=ax, secondary_y=True, title=name)
    ax2.set_ylim(0)

fig.tight_layout()

numerical_label = ['P' + str(i) for i in range(1,38)]
trainData['age'] = (datetime.now() - trainData['Open Date']).astype('timedelta64[D]') / 365   
trainData['Type'] = LabelEncoder().fit_transform(trainData['Type'])
trainData['City Group'] = LabelEncoder().fit_transform(trainData['City Group'])

testData['age'] = (datetime.now() - testData['Open Date']).astype('timedelta64[D]') / 365   
testData['Type'] = LabelEncoder().fit_transform(testData['Type'])
testData['City Group'] = LabelEncoder().fit_transform(testData['City Group'])

X_names = numerical_label + ['age', 'Type', 'City Group']
clf=RandomForestRegressor(n_estimators=1000, max_features=2)
X_train = trainData[X_names]
y_train = trainData['revenue']

clf.fit(X_train, y_train)

X_test = testData[X_names]
pred = clf.predict(X_test)
pred

closs = clf.score(X_train, y_train)
closs

result = pd.read_csv(data_dir + 'sampleSubmission.csv')
result['Prediction'] = pred
result.to_csv('output.csv',index=False)