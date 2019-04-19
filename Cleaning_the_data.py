import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

import warnings

warnings.filterwarnings('ignore')

train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")

passengerid = test.PassengerId
train.drop(['PassengerId'], axis=1, inplace=True)
test.drop(['PassengerId'], axis=1, inplace=True)

# ==============================================================================
# #查看缺失值
# total = train.isnull().sum().sort_values(ascending = False)
# percent = round(train.isnull().sum().sort_values(ascending = False)/len(train)*100, 2)
# pd.concat([total, percent], axis = 1,keys= ['Total', 'Percent'])
#
# total = test.isnull().sum().sort_values(ascending = False)
# percent = round(test.isnull().sum().sort_values(ascending = False)/len(test)*100, 2)
# pd.concat([total, percent], axis = 1,keys= ['Total', 'Percent'])
#
# percent = pd.DataFrame(round(train.Embarked.value_counts(dropna=False, normalize=True)*100,2))
# total = pd.DataFrame(train.Embarked.value_counts(dropna=False))
# total.columns = ["Total"]
# percent.columns = ['Percent']
# pd.concat([total, percent], axis = 1)
#
# ==============================================================================

# ==============================================================================
# #Embarked缺失了在训练集缺失了两个数据
# fig, ax = plt.subplots(figsize=(16,12),ncols=2)
# ax1 = sns.boxplot(x="Embarked", y="Fare", hue="Pclass", data=train, ax = ax[0]);
# ax2 = sns.boxplot(x="Embarked", y="Fare", hue="Pclass", data=test, ax = ax[1]);
# ax1.set_title("Training Set", fontsize = 18)
# ax2.set_title('Test Set',  fontsize = 18)
# fig.show()
# ==============================================================================
# 作者画图分析，联合Fare和Pclass分析得出最好的是C
train.Embarked.fillna("C", inplace=True)

# Cabin特征，没有直接删除，根据分析，可能有无会反映地位，因此保留。
train.Cabin.fillna("N", inplace=True)
test.Cabin.fillna("N", inplace=True)

train.Cabin = [i[0] for i in train.Cabin]
test.Cabin = [i[0] for i in test.Cabin]

# 测试集的Fare缺失了一个数据，联合几列特征用平均值填充
test[test.Fare.isnull()]
missing_value = test[(test.Pclass == 3) & (test.Embarked == "S") & (test.Sex == "male")].Fare.mean()
test.Fare.fillna(missing_value, inplace=True)

# ==============================================================================
# #接下来是一系列的画图可视化，我列一个比较新奇的
# #票价与存活率之间的分析
# fig = plt.figure(figsize=(15,8),)
# ax=sns.kdeplot(train.loc[(train['Survived'] == 0),'Fare'] , color='gray',shade=True,label='not survived')
# ax=sns.kdeplot(train.loc[(train['Survived'] == 1),'Fare'] , color='g',shade=True, label='survived')
# plt.title('Fare Distribution Survived vs Non Survived', fontsize = 25)
# plt.ylabel("Frequency of Passenger Survived", fontsize = 15)
# plt.xlabel("Fare", fontsize = 15)
#
# train[train.Fare > 280]
# #看起来票价512是一个异常值，但是我们暂时保留
#
#
# #接下来作者做了相互几个特征之间的联合关系，代码值得记录学习
# #性别，年龄和存活率
# pal = {1:"seagreen", 0:"gray"}
# g = sns.FacetGrid(train,size=5, col="Sex", row="Survived", margin_titles=True, hue = "Survived",
#                   palette=pal)
# g = g.map(plt.hist, "Age", edgecolor = 'white');
# g.fig.suptitle("Survived by Sex and Age", size = 25)
# plt.subplots_adjust(top=0.90)
#
# #性别，年龄，登船港口和存活率
# g = sns.FacetGrid(train,size=5, col="Sex", row="Embarked", margin_titles=True, hue = "Survived",
#                   palette = pal
#                   )
# g = g.map(plt.hist, "Age", edgecolor = 'white').add_legend();
# g.fig.suptitle("Survived by Sex and Age", size = 25)
# plt.subplots_adjust(top=0.90)
#
# #性别，票价，年龄和存活率
# g = sns.FacetGrid(train, size=5,hue="Survived", col ="Sex", margin_titles=True,
#                 palette=pal,)
# g.map(plt.scatter, "Fare", "Age",edgecolor="w").add_legend()
# g.fig.suptitle("Survived by Sex, Fare and Age", size = 25)
# plt.subplots_adjust(top=0.85)
# ==============================================================================
# 认为512为异常值，应该删除
train = train[train.Fare < 500]

train['Sex'] = train.Sex.apply(lambda x: 0 if x == "female" else 1)
test['Sex'] = test.Sex.apply(lambda x: 0 if x == "female" else 1)

# ==============================================================================
# #一致性
# pd.DataFrame(abs(train.corr()['Survived']).sort_values(ascending = False))
#
# corr = train.corr()**2
# corr.Survived.sort_values(ascending=False)
# #平方后放大了差异
# ==============================================================================


# 特征工程
# name_length
train['name_length'] = [len(i) for i in train.Name]
test['name_length'] = [len(i) for i in test.Name]


def name_length_group(size):
    a = ''
    if (size <= 20):
        a = 'short'
    elif (size <= 35):
        a = 'medium'
    elif (size <= 45):
        a = 'good'
    else:
        a = 'long'
    return a


train['nLength_group'] = train['name_length'].map(name_length_group)
test['nLength_group'] = test['name_length'].map(name_length_group)

# title
train["title"] = [i.split('.')[0] for i in train.Name]
train["title"] = [i.split(',')[1] for i in train.title]
test["title"] = [i.split('.')[0] for i in test.Name]
test["title"] = [i.split(',')[1] for i in test.title]
train["title"] = [i.replace('Ms', 'Miss') for i in train.title]
train["title"] = [i.replace('Mlle', 'Miss') for i in train.title]
train["title"] = [i.replace('Mme', 'Mrs') for i in train.title]
train["title"] = [i.replace('Dr', 'rare') for i in train.title]
train["title"] = [i.replace('Col', 'rare') for i in train.title]
train["title"] = [i.replace('Major', 'rare') for i in train.title]
train["title"] = [i.replace('Don', 'rare') for i in train.title]
train["title"] = [i.replace('Jonkheer', 'rare') for i in train.title]
train["title"] = [i.replace('Sir', 'rare') for i in train.title]
train["title"] = [i.replace('Lady', 'rare') for i in train.title]
train["title"] = [i.replace('Capt', 'rare') for i in train.title]
train["title"] = [i.replace('the Countess', 'rare') for i in train.title]
train["title"] = [i.replace('Rev', 'rare') for i in train.title]
test['title'] = [i.replace('Ms', 'Miss') for i in test.title]
test['title'] = [i.replace('Dr', 'rare') for i in test.title]
test['title'] = [i.replace('Col', 'rare') for i in test.title]
test['title'] = [i.replace('Dona', 'rare') for i in test.title]
test['title'] = [i.replace('Rev', 'rare') for i in test.title]

# has_cabin
train["has_cabin"] = [0 if i == 'N' else 1 for i in train.Cabin]
test["has_cabin"] = [0 if i == 'N' else 1 for i in test.Cabin]

# child
train['child'] = [1 if i < 16 else 0 for i in train.Age]
test['child'] = [1 if i < 16 else 0 for i in test.Age]

# family
train['family_size'] = train.SibSp + train.Parch + 1
test['family_size'] = test.SibSp + test.Parch + 1


def family_group(size):
    a = ''
    if (size <= 1):
        a = 'loner'
    elif (size <= 4):
        a = 'small'
    else:
        a = 'large'
    return a


train['family_group'] = train['family_size'].map(family_group)
test['family_group'] = test['family_size'].map(family_group)

# is_alone
train['is_alone'] = [1 if i < 2 else 0 for i in train.family_size]
test['is_alone'] = [1 if i < 2 else 0 for i in test.family_size]

# calculated_fare feature
train['calculated_fare'] = train.Fare / train.family_size
test['calculated_fare'] = test.Fare / test.family_size


# fare_group
def fare_group(fare):
    a = ''
    if fare <= 4:
        a = 'Very_low'
    elif fare <= 10:
        a = 'low'
    elif fare <= 20:
        a = 'mid'
    elif fare <= 45:
        a = 'high'
    else:
        a = "very_high"
    return a


train['fare_group'] = train['calculated_fare'].map(fare_group)
test['fare_group'] = test['calculated_fare'].map(fare_group)

# dummy
train = pd.get_dummies(train,
                       columns=['title', "Pclass", 'Cabin', 'Embarked', 'nLength_group', 'family_group', 'fare_group'],
                       drop_first=True)
test = pd.get_dummies(test,
                      columns=['title', "Pclass", 'Cabin', 'Embarked', 'nLength_group', 'family_group', 'fare_group'],
                      drop_first=True)
train.drop(['Cabin_T', 'family_size', 'Ticket', 'Name', 'Fare'], axis=1, inplace=True)
test.drop(['Ticket', 'Name', 'family_size', "Fare"], axis=1, inplace=True)

train = pd.concat([train[["Survived", "Age", "Sex"]], train.loc[:, "SibSp":]], axis=1)
test = pd.concat([test[["Age", "Sex"]], test.loc[:, "SibSp":]], axis=1)

# 用随机森林填充年龄的缺失值
from sklearn.ensemble import RandomForestRegressor


def completing_age(df):
    age_df = df.loc[:, "Age":]
    temp_train = age_df.loc[age_df.Age.notnull()]  ## df with age values
    temp_test = age_df.loc[age_df.Age.isnull()]  ## df without age values

    y = temp_train.Age.values  ## setting target variables(age) in y
    x = temp_train.loc[:, "Sex":].values

    rfr = RandomForestRegressor(n_estimators=1500, n_jobs=-1)
    rfr.fit(x, y)
    predicted_age = rfr.predict(temp_test.loc[:, "Sex":])
    df.loc[df.Age.isnull(), "Age"] = predicted_age

    return df

print("Train age missing value: " + str((train.Age.isnull().sum()/len(train))*100)+str("%"))
print("Test age missing value: " + str((test.Age.isnull().sum()/len(test))*100)+str("%"))
train = completing_age(train)
test = completing_age(test)
print("Train age missing value: " + str((train.Age.isnull().sum()/len(train))*100)+str("%"))
print("Test age missing value: " + str((test.Age.isnull().sum()/len(test))*100)+str("%"))

