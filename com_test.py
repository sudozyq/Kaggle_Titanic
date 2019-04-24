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


train = completing_age(train)
test = completing_age(test)

X = train.drop(['Survived'], axis=1)
y = train["Survived"]

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=.33, random_state=0)

# 特征缩放
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

test = sc.transform(test)

from sklearn.model_selection import StratifiedKFold

# 逻辑回归
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_test)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator=logreg, X=x_train, y=y_train, cv=10, n_jobs=-1)
logreg_accy = accuracies.mean()
print(round((logreg_accy), 3))  # 0.833

# 网格搜索
C_vals = [0.099, 0.1, 0.2, 0.5, 12, 13, 14, 15, 16, 16.5, 17, 17.5, 18]
penalties = ['l1', 'l2']

param = {'penalty': penalties,
         'C': C_vals
         }
grid_search = GridSearchCV(estimator=logreg,
                           param_grid=param,
                           scoring='accuracy',
                           cv=10
                           )

grid_search = grid_search.fit(x_train, y_train)
print(grid_search.best_params_)
print(grid_search.best_score_)  # 0.841750841751

logreg_grid = grid_search.best_estimator_
logreg_accy = logreg_grid.score(x_test, y_test)
logreg_accy  # 0.81292517006802723

# KNN
from sklearn.neighbors import KNeighborsClassifier

nn_scores = []
best_prediction = [-1, -1]
for i in range(1, 100):
    knn = KNeighborsClassifier(n_neighbors=i, weights='distance', metric='minkowski', p=2)
    knn.fit(x_train, y_train)
    score = accuracy_score(y_test, knn.predict(x_test))
    # print i, score
    if score > best_prediction[1]:
        best_prediction = [i, score]
    nn_scores.append(score)

print(best_prediction)
plt.plot(range(1, 100), nn_scores)  # [22, 0.80612244897959184]

# 网格搜索
knn = KNeighborsClassifier()
n_neighbors = [17, 18, 19, 20, 21, 22, 23, 24, 25]
weights = ['uniform', 'distance']
param = {'n_neighbors': n_neighbors,
         'weights': weights}
grid2 = GridSearchCV(knn,
                     param,
                     verbose=False,
                     cv=StratifiedKFold(n_splits=5, random_state=15, shuffle=True)
                     )
grid2.fit(x_train, y_train)

print(grid2.best_params_)  # {'n_neighbors': 17, 'weights': 'uniform'}
print(grid2.best_score_)  # 0.819865319865

knn_grid = grid2.best_estimator_
knn_accy = knn_grid.score(x_test, y_test)
knn_accy  # 0.79251700680272108

# 高斯朴素贝叶斯
from sklearn.naive_bayes import GaussianNB

gaussian = GaussianNB()
gaussian.fit(x_train, y_train)
y_pred = gaussian.predict(x_test)
gaussian_accy = round(accuracy_score(y_pred, y_test), 3)
print(gaussian_accy)  # 0.765

# SVM
from sklearn.svm import SVC

svc = SVC(kernel='rbf', probability=True, random_state=1, C=3)
svc.fit(x_train, y_train)
y_pred = svc.predict(x_test)
svc_accy = round(accuracy_score(y_pred, y_test), 3)
print(svc_accy)  # 0.806

# 决策树
dectree = DecisionTreeClassifier()
dectree.fit(x_train, y_train)
y_pred = dectree.predict(x_test)
dectree_accy = round(accuracy_score(y_pred, y_test), 3)
print(dectree_accy)  # 0.707

max_depth = range(1, 30)
max_feature = [21, 22, 23, 24, 25, 26, 'auto']
criterion = ["entropy", "gini"]

param = {'max_depth': max_depth,
         'max_features': max_feature,
         'criterion': criterion}
decisiontree_grid = GridSearchCV(dectree,
                                 param_grid=param,
                                 verbose=False,
                                 cv=StratifiedKFold(n_splits=20, random_state=15, shuffle=True),
                                 n_jobs=-1)
decisiontree_grid.fit(x_train, y_train)

print(decisiontree_grid.best_params_)  # {'max_depth': 4, 'max_features': 23, 'criterion': 'entropy'}
print(decisiontree_grid.best_score_)  # 0.843434343434
decisiontree_grid = decisiontree_grid.best_estimator_
decisiontree_grid.score(x_test, y_test)  # 0.80272108843537415

# Bagging Classifier
from sklearn.ensemble import BaggingClassifier

BaggingClassifier = BaggingClassifier()
BaggingClassifier.fit(x_train, y_train)
y_pred = BaggingClassifier.predict(x_test)
bagging_accy = round(accuracy_score(y_pred, y_test), 3)
print(bagging_accy)  # 0.793

# 随机森林
from sklearn.ensemble import RandomForestClassifier

randomforest = RandomForestClassifier(n_estimators=100, max_depth=9, min_samples_split=6, min_samples_leaf=4)
randomforest.fit(x_train, y_train)
y_pred = randomforest.predict(x_test)
random_accy = round(accuracy_score(y_pred, y_test), 3)
print(random_accy)  # 0.823

n_estimators = [100, 120]
max_depth = range(1, 30)

parameters = {'n_estimators': n_estimators,
              'max_depth': max_depth,
              }
randomforest_grid = GridSearchCV(randomforest,
                                 param_grid=parameters,
                                 cv=StratifiedKFold(n_splits=20, random_state=15, shuffle=True),
                                 n_jobs=-1
                                 )
randomforest_grid.fit(x_train, y_train)
randomforest_grid.score(x_test, y_test)  # 0.8231292517006803

# GBDT
from sklearn.ensemble import GradientBoostingClassifier

gradient = GradientBoostingClassifier()
gradient.fit(x_train, y_train)
y_pred = gradient.predict(x_test)
gradient_accy = round(accuracy_score(y_pred, y_test), 3)
print(gradient_accy)  # 0.81

# XGB
from xgboost import XGBClassifier

XGBClassifier = XGBClassifier()
XGBClassifier.fit(x_train, y_train)
y_pred = XGBClassifier.predict(x_test)
XGBClassifier_accy = round(accuracy_score(y_pred, y_test), 3)
print(XGBClassifier_accy)  # 0.816

# AdaBoost Classifier
from sklearn.ensemble import AdaBoostClassifier

adaboost = AdaBoostClassifier()
adaboost.fit(x_train, y_train)
y_pred = adaboost.predict(x_test)
adaboost_accy = round(accuracy_score(y_pred, y_test), 3)
print(adaboost_accy)  # 0.786

# Extra Trees Classifier
from sklearn.ensemble import ExtraTreesClassifier

ExtraTreesClassifier = ExtraTreesClassifier()
ExtraTreesClassifier.fit(x_train, y_train)
y_pred = ExtraTreesClassifier.predict(x_test)
extraTree_accy = round(accuracy_score(y_pred, y_test), 3)
print(extraTree_accy)  # 0.786

# Gaussian Process Classifier
from sklearn.gaussian_process import GaussianProcessClassifier

GaussianProcessClassifier = GaussianProcessClassifier()
GaussianProcessClassifier.fit(x_train, y_train)
y_pred = GaussianProcessClassifier.predict(x_test)
gau_pro_accy = round(accuracy_score(y_pred, y_test), 3)
print(gau_pro_accy)  # 0.786

# 投票法
from sklearn.ensemble import VotingClassifier

voting_classifier = VotingClassifier(estimators=[
    ('logreg_grid', logreg_grid),
    ('logreg', logreg),
    ('svc', svc),
    ('random_forest', randomforest),
    ('gradient_boosting', gradient),
    ('decision_tree', dectree),
    ('decision_tree_grid', decisiontree_grid),
    ('knn', knn),
    ('knn_grid', knn_grid),
    ('XGB Classifier', XGBClassifier),
    ('BaggingClassifier', BaggingClassifier),
    ('ExtraTreesClassifier', ExtraTreesClassifier),
    ('gaussian', gaussian),
    ('gaussian process classifier', GaussianProcessClassifier)], voting='soft')

voting_classifier = voting_classifier.fit(x_train, y_train)

y_pred = voting_classifier.predict(x_test)
voting_accy = round(accuracy_score(y_pred, y_test), 3)
print(voting_accy)  # 0.833

models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression',
              'Random Forest', 'Naive Bayes',
              'Decision Tree', 'Gradient Boosting Classifier', 'Voting Classifier', 'XGB Classifier',
              'ExtraTrees Classifier', 'Bagging Classifier'],
    'Score': [svc_accy, knn_accy, logreg_accy,
              random_accy, gaussian_accy, dectree_accy,
              gradient_accy, voting_accy, XGBClassifier_accy, extraTree_accy, bagging_accy]})
models.sort_values(by='Score', ascending=False)

test_prediction = voting_classifier.predict(test)
submission = pd.DataFrame({
    "PassengerId": passengerid,
    "Survived": test_prediction
})

submission.PassengerId = submission.PassengerId.astype(int)
submission.Survived = submission.Survived.astype(int)

submission.to_csv("titanic1_submission.csv", index=False)