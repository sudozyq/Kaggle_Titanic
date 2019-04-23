import random
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import warnings
import scipy.stats as stats
import os
# 导入随机森林库
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
warnings.filterwarnings('ignore')
# 让matplotlib
plt.rcParams['font.sans-serif'] = [u'SimHei']  # 设定中文黑体 FangSong/KaiTi
plt.rcParams['axes.unicode_minus'] = False

###################################################
#                   pandas基础                     #
###################################################

# # 打印路径下的文件名
# print(os.listdir("./data/"))

# 导入数据集
train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")

# #随机预览5条数据集
# print(train.sample(5))
# print(test.sample(5))

# # 数据集基本信息
# print("The shape of the train data is (row, column):"+ str(train.shape))
# print(train.info())
# print("The shape of the test data is (row, column):"+ str(test.shape))
# print(test.info())

# # 数据可视化
#
# fig = plt.figure()
# fig.set(alpha=0.2)  # 设定图表颜色alpha参数
#
# plt.subplot2grid([2, 3], [0, 0])                # 将整个图像窗口分为2行3列, 当前位置[0, 0]
# train.Survived.value_counts().plot(kind='bar')  # 柱状图
# plt.title(u"获救情况 (1为获救)")                   # 标题 U代表unicode
# plt.ylabel(u"人数")                              # 设定纵坐标名称
#
# plt.subplot2grid((2, 3), (0, 1))
# train.Pclass.value_counts().plot(kind="bar")
# plt.ylabel(u"人数")
# plt.title(u"乘客等级分布")
#
# plt.subplot2grid((2, 3), (0, 2))
# plt.scatter(train.Survived, train.Age)
# plt.ylabel(u"年龄")
# plt.grid(b=True, which='major', axis='y')
# plt.title(u"按年龄看获救分布 (1为获救)")
#
# plt.subplot2grid((2, 3), (1, 0), colspan=2)    # 将整个图像窗口分为2行3列, 当前位置[1, 0], 图像占2列
# train.Age[train.Pclass == 1].plot(kind='kde')  # 连续曲线
# train.Age[train.Pclass == 2].plot(kind='kde')
# train.Age[train.Pclass == 3].plot(kind='kde')
# plt.xlabel(u"年龄")
# plt.ylabel(u"密度")
# plt.title(u"各等级的乘客年龄分布")
# plt.legend((u'头等舱', u'2等舱', u'3等舱'), loc='best')  # sets our legend for our graph.
#
# plt.subplot2grid((2, 3), (1, 2))
# train.Embarked.value_counts().plot(kind='bar')
# plt.title(u"各登船口岸上船人数")
# plt.ylabel(u"人数")
# plt.show()

# 总览与数据清洗
passengerid = test.PassengerId

# 去除ID与票号这种无关数据
train.drop(['PassengerId'], axis=1, inplace=True)
test.drop(['PassengerId'], axis=1, inplace=True)

# print(train.info())
# print("="*20 + "我是分隔线" + "="*20)
# print(test.info())


# # 训练集的缺失值并列出百分比
# total = train.isnull().sum().sort_values(ascending=False)
# percent = round(train.isnull().sum().sort_values(ascending=False)/len(train)*100, 2)
# print(pd.concat([total, percent], axis=1, keys=['Total', 'Percent']))
#
# print("="*20 + "我是分隔线" + "="*20)
#
# # 测试集的缺失值并列出百分比
# total = test.isnull().sum().sort_values(ascending=False)
# percent = round(test.isnull().sum().sort_values(ascending=False)/len(test)*100, 2)
# print(pd.concat([total, percent], axis=1, keys=['Total', 'Percent']))


# # 乘船地点的特点
# percent = pd.DataFrame(round(train.Embarked.value_counts(dropna=False, normalize=True)*100, 2))
# # creating a df with th
# total = pd.DataFrame(train.Embarked.value_counts(dropna=False))
# # 连接显示百分比与总数
# total.columns = ["Total"]
# percent.columns = ['Percent']
# print(pd.concat([total, percent], axis=1))      # axis = 0按行拼接，axis = 1案列拼接
#
# # 输出其中的空值
# print(train[train.Embarked.isnull()])
#
# # 通过其他值猜出空值
# fig, ax = plt.subplots(figsize=(16, 12), ncols=2)
# ax1 = sns.boxplot(x="Embarked", y="Fare", hue="Pclass", data=train, ax=ax[0])
# ax2 = sns.boxplot(x="Embarked", y="Fare", hue="Pclass", data=test, ax=ax[1])
# ax1.set_title("训练集", fontsize=18)
# ax2.set_title("测试集",  fontsize=18)
# plt.show()

# 根据上图因为C地点登陆的人票价最接近80，所以填入C
train.Embarked.fillna("C", inplace=True)

# # 客舱值的特点
# print("Train Cabin missing: " + str(train.Cabin.isnull().sum()/len(train.Cabin)))
# print("Test Cabin missing: " + str(test.Cabin.isnull().sum()/len(test.Cabin)))
# 结果看到78%的值已缺失

# Cabin特征，没有直接删除，根据分析，可能有无会反映地位，因此保留。
train.Cabin.fillna("N", inplace=True)
test.Cabin.fillna("N", inplace=True)

train.Cabin = [i[0] for i in train.Cabin]
test.Cabin = [i[0] for i in test.Cabin]

# # # 数据清洗
# # # Concat train and test into a variable "all_data"
# survivers = train.Survived
# train.drop(["Survived"], axis=1, inplace=True)
# all_data = pd.concat([train, test], ignore_index=False)
# # # 把缺失值填入 N
# all_data.Cabin.fillna("N", inplace=True)
# all_data.Cabin = [i[0] for i in all_data.Cabin]
# #
# # # 利用groupby函数对每个cabin分类
# with_N = all_data[all_data.Cabin == "N"]
# without_N = all_data[all_data.Cabin != "N"]
# # print(all_data.groupby("Cabin")['Fare'].mean().sort_values())
# #
# # 利用mean值分类cabin值
# def cabin_estimator(i):
#     a = 0
#     if i < 16:
#         a = "G"
#     elif i >= 16 and i<27:
#         a = "F"
#     elif i>=27 and i<38:
#         a = "T"
#     elif i>=38 and i<47:
#         a = "A"
#     elif i>= 47 and i<53:
#         a = "E"
#     elif i>= 53 and i<54:
#         a = "D"
#     elif i>=54 and i<116:
#         a = 'C'
#     else:
#         a = "B"
#     return a
#
#
# with_N['Cabin'] = with_N.Fare.apply(lambda x: cabin_estimator(x))
# #
# # # 修改训练集
# all_data = pd.concat([with_N, without_N], axis=0)
# # # 用PassengerId分割训练集与测试集
# all_data.sort_values(by='PassengerId', inplace=True)
# # # 分割
# train = all_data[:891]
# test = all_data[891:]
# # # 把训练集的生还者加入获救标签
# train['Survived'] = survivers


# # 票价特点分析
# # 找出空票价的数据
# print(test[test.Fare.isnull()])

test[test.Fare.isnull()]
missing_value = test[(test.Pclass == 3) & (test.Embarked == "S") & (test.Sex == "male")].Fare.mean()
test.Fare.fillna(missing_value, inplace=True)

#
# # 根据Pclass信息，性别，登陆地点填入值
# missing_value = test[(test.Pclass == 3) & (test.Embarked == "S") & (test.Sex == "male")].Fare.mean()
# # 填充
# test.Fare.fillna(missing_value, inplace=True)
#
# # 年龄特点分析
# # 显示年龄数据的缺失率
# print("Train age missing value: " + str((train.Age.isnull().sum()/len(train))*100)+str("%"))
# print("Test age missing value: " + str((test.Age.isnull().sum()/len(test))*100)+str("%"))
# # 可以看到有20%的缺失率
# # 使用机器学习的方式填充


##############################
#      数据可视化与特征联系      #
##############################

# # 性别与存活的关系可视化
# pal = {'male':"green", 'female':"Pink"}
# plt.subplots(figsize=(15, 8))
# ax = sns.barplot(x="Sex",
#             y="Survived",
#             data=train,
#             palette=pal,
#             linewidth=2)
# plt.title("存活/死亡 乘客的性别分布", fontsize=25)
# plt.ylabel("% of passenger survived", fontsize=15)
# plt.xlabel("Sex", fontsize=15)
# plt.show()

# # 性别与存活的关系可视化2
# pal = {1:"seagreen", 0:"gray"}
# sns.set(style="darkgrid")
# plt.subplots(figsize=(15, 8))
# ax = sns.countplot(x="Sex",
#                    hue="Survived",
#                    data=train,
#                    linewidth=2,
#                    palette=pal
# )
#
# ## Fixing title, xlabel and ylabel
# plt.title("乘客性别分布 - 生还与死亡", fontsize=25)
# plt.xlabel("Sex", fontsize=15)
# plt.ylabel("# of Passenger Survived", fontsize=15)
#
# ## Fixing xticks
# #labels = ['Female', 'Male']
# #plt.xticks(sorted(train.Sex.unique()), labels)
#
# ## Fixing legends
# leg = ax.get_legend()
# leg.set_title("Survived")
# legs = leg.texts
# legs[0].set_text("No")
# legs[1].set_text("Yes")
# plt.show()

# # 仓位等级与存活率
# plt.subplots(figsize=(15, 10))
# sns.barplot(x="Pclass",
#             y="Survived",
#             data=train,
#             linewidth=2)
# plt.title("乘客仓位分布 - 存活与死亡", fontsize = 25)
# plt.xlabel("Socio-Economic class", fontsize = 15);
# plt.ylabel("% of Passenger Survived", fontsize = 15);
# labels = ['Upper', 'Middle', 'Lower']
# #val = sorted(train.Pclass.unique())
# val = [0, 1, 2] ## this is just a temporary trick to get the label right.
# plt.xticks(val, labels)
# plt.show()

# # 密度图
# # 仓位等级与存活率
# fig = plt.figure(figsize=(15, 8),)
# ## I have included to different ways to code a plot below, choose the one that suites you.
# ax = sns.kdeplot(train.Pclass[train.Survived == 0],
#                color='gray',
#                shade=True,
#                label='not survived')
# ax = sns.kdeplot(train.loc[(train['Survived'] == 1), 'Pclass'],
#                color='g',
#                shade=True,
#                label='survived')
# plt.title('乘客仓位分布 - 存活与死亡', fontsize=25)
# plt.ylabel("Frequency of Passenger Survived", fontsize=15)
# plt.xlabel("Passenger Class", fontsize=15)
# ## Converting xticks into words for better understanding
# labels = ['Upper', 'Middle', 'Lower']
# plt.xticks(sorted(train.Pclass.unique()), labels)
# plt.show()

# # 票价与存活
# # 密度图
# fig = plt.figure(figsize=(15, 8),)
# ax = sns.kdeplot(train.loc[(train['Survived'] == 0), 'Fare'], color='gray', shade=True, label='not survived')
# ax = sns.kdeplot(train.loc[(train['Survived'] == 1), 'Fare'], color='g', shade=True, label='survived')
# plt.title('票价分布 与 存活 - 死亡', fontsize=25)
# plt.ylabel("Frequency of Passenger Survived", fontsize=15)
# plt.xlabel("Fare", fontsize=15)
# plt.show()
#
# print(train[train.Fare > 280])

# # 年龄与存活
# # 密度图
# fig = plt.figure(figsize=(15, 8),)
# ax = sns.kdeplot(train.loc[(train['Survived'] == 0), 'Age'], color='gray', shade=True, label='not survived')
# ax = sns.kdeplot(train.loc[(train['Survived'] == 1), 'Age'], color='g', shade=True, label='survived')
# plt.title('年龄分布 - 存活与死亡', fontsize=25)
# plt.xlabel("Age", fontsize=15)
# plt.ylabel('Frequency', fontsize=15)
# plt.show()

# # 多个特征的联系
# # 存活，性别，年龄
# pal = {1: "seagreen", 0: "gray"}
# g = sns.FacetGrid(train, size=5, col="Sex", row="Survived", margin_titles=True, hue="Survived",
#                   palette=pal)
# g = g.map(plt.hist, "Age", edgecolor='white')
# g.fig.suptitle("存活与年龄性别", size=25)
# plt.subplots_adjust(top=0.90)
# plt.show()

# # 存活，性别，年龄， 仓位
# pal = {1:"seagreen", 0:"gray"}
# g = sns.FacetGrid(train, size=5, col="Sex", row="Embarked", margin_titles=True, hue = "Survived",
#                   palette=pal
#                   )
# g = g.map(plt.hist, "Age", edgecolor = 'white').add_legend()
# g.fig.suptitle("存活 - 性别 - 年龄", size = 25)
# plt.subplots_adjust(top=0.90)
# plt.show()

# # 存活， 性别，年龄，票价
# pal = {1: "seagreen", 0: "gray"}
# g = sns.FacetGrid(train, size=5, hue="Survived", col="Sex", margin_titles=True,
#                 palette=pal,)
# g.map(plt.scatter, "Fare", "Age", edgecolor="w").add_legend()
# g.fig.suptitle("存活 - 性别 - 票价 - 年龄", size=25)
# plt.subplots_adjust(top=0.85)
# plt.show()

# # 数据清洗
# # 删除票价大于500的数值
# train = train[train.Fare < 500]
# # 因子图
# sns.factorplot(x="Parch", y="Survived", data=train, kind="point", size=8)
# plt.title("Factorplot of Parents/Children survived", fontsize=25)
# plt.subplots_adjust(top=0.85)
# plt.show()
# # 可见带一大家子的存活率比较低

# sns.factorplot(x =  "SibSp", y = "Survived", data = train,kind = "point",size = 8)
# plt.title('Factorplot of Sibilings/Spouses survived', fontsize = 25)
# plt.subplots_adjust(top=0.85)
# plt.show()
# 可见单身狗或者较少成员一起出行的人存活率高

# # 填入数据集，0代表女性，1代表男性
# train['Sex'] = train.Sex.apply(lambda x: 0 if x == "female" else 1)
# test['Sex'] = test.Sex.apply(lambda x: 0 if x == "female" else 1)

#######################################
#             统计概况                  #
#######################################
# print(train.describe())
# print("===================分割线===============")
# print(train.describe(include=['O']))
# print("===================分割线===============")
# print(train[['Pclass', 'Survived']].groupby("Pclass").mean().reset_index())
# print("===================分割线===============")
# # 总览一下存活数据
# survived_summary = train.groupby("Survived")
# print(survived_summary.mean().reset_index())
# print("===================分割线===============")
# # 性别数据
# survived_summary = train.groupby("Sex")
# print(survived_summary.mean().reset_index())
# print("===================分割线===============")
# # 仓位数据
# survived_summary = train.groupby("Pclass")
# print(survived_summary.mean().reset_index())

###################################
#          相关性矩阵与热力图         #
###################################
# 删除票价大于500的数值
train = train[train.Fare < 500]
# 填入数据集，0代表女性，1代表男性
train['Sex'] = train.Sex.apply(lambda x: 0 if x == "female" else 1)
test['Sex'] = test.Sex.apply(lambda x: 0 if x == "female" else 1)
# 相关性
# print(pd.DataFrame(abs(train.corr()['Survived']).sort_values(ascending=False)))
# print("===================分割线===============")
# 找出相关性最高的因素
# 将相关特征平方化不仅可以得到正相关，而且可以放大相关关系
# corr = train.corr()**2
# print(corr.Survived.sort_values(ascending=False))

# # 同热力图查看特点之间的相关性
# # 上三角矩阵
# mask = np.zeros_like(train.corr(), dtype=np.bool)
# # mask[np.triu_indices_from(mask)] = True
# plt.subplots(figsize=(15, 12))
# sns.heatmap(train.corr(),
#             annot=True,
#             #mask = mask,
#             cmap='RdBu_r',
#             linewidths=0.1,
#             linecolor='white',
#             vmax=.9,
#             square=True)
# plt.title("特征之间的相关性", y=1.03, fontsize=20)
# plt.show()

# # 相关性检验
# male_mean = train[train['Sex'] == 1].Survived.mean()
# female_mean = train[train['Sex'] == 0].Survived.mean()
# print("Male survival mean: " + str(male_mean))
# print("female survival mean: " + str(female_mean))
# print("The mean difference between male and female survival rate: " + str(female_mean - male_mean))
#
# # 划分不同性别数据集
# male = train[train['Sex'] == 1]
# female = train[train['Sex'] == 0]
# #
# # # 随机抽取50个值
# male_sample = random.sample(list(male['Survived']), 50)
# female_sample = random.sample(list(female['Survived']), 50)
# #
# # Taking a sample means of survival feature from male and female
# male_sample_mean = np.mean(male_sample)
# female_sample_mean = np.mean(female_sample)
#
# # Print them out
# print("Male sample mean: " + str(male_sample_mean))
# print("Female sample mean: " + str(female_sample_mean))
# print("Difference between male and female sample mean: " + str(female_sample_mean - male_sample_mean))

# # t检验
# print(stats.ttest_ind(male_sample, female_sample))
# print("This is the p-value when we break it into standard form: " +
#       format(stats.ttest_ind(male_sample, female_sample).pvalue, '.32f'))

######################################
#              特征工程                #
######################################
# # 创建新列
train['name_length'] = [len(i) for i in train.Name]
test['name_length'] = [len(i) for i in test.Name]


def name_length_group(size):
    a = ''
    if (size <=20):
        a = 'short'
    elif (size <=35):
        a = 'medium'
    elif (size <=45):
        a = 'good'
    else:
        a = 'long'
    return a


train['nLength_group'] = train['name_length'].map(name_length_group)
test['nLength_group'] = test['name_length'].map(name_length_group)

# 从名字处拿到标题
train["title"] = [i.split('.')[0] for i in train.Name]
train["title"] = [i.split(',')[1] for i in train.title]
test["title"] = [i.split('.')[0] for i in test.Name]
test["title"] = [i.split(',')[1] for i in test.title]

# # rare_title = ['the Countess','Capt','Lady','Sir','Jonkheer','Don','Major','Col']
# # train.Name = ['rare' for i in train.Name for j in rare_title if i == j]
# # 训练数据
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

# # rare_title = ['the Countess','Capt','Lady','Sir','Jonkheer','Don','Major','Col']
# # train.Name = ['rare' for i in train.Name for j in rare_title if i == j]
# 训练数据
test['title'] = [i.replace('Ms', 'Miss') for i in test.title]
test['title'] = [i.replace('Dr', 'rare') for i in test.title]
test['title'] = [i.replace('Col', 'rare') for i in test.title]
test['title'] = [i.replace('Dona', 'rare') for i in test.title]
test['title'] = [i.replace('Rev', 'rare') for i in test.title]

# famliy size 特点
train['family_size'] = train.SibSp + train.Parch+1
test['family_size'] = test.SibSp + test.Parch+1


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

# 独立特征
train['is_alone'] = [1 if i<2 else 0 for i in train.family_size]
test['is_alone'] = [1 if i<2 else 0 for i in test.family_size]

# 票价特征
# print(train.Ticket.value_counts().sample(10))

train.drop(['Ticket'], axis=1, inplace=True)
test.drop(['Ticket'], axis=1, inplace=True)

# 票价特征
# 根据家庭人数计算票价
train['calculated_fare'] = train.Fare/train.family_size
test['calculated_fare'] = test.Fare/test.family_size


def fare_group(fare):
    a= ''
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

# 创建虚拟变量 比如：male -> 1, female -> 0
train = pd.get_dummies(train, columns=['title', "Pclass", 'Cabin', 'Embarked', 'nLength_group', 'family_group', 'fare_group'], drop_first=False)
test = pd.get_dummies(test, columns=['title', "Pclass", 'Cabin', 'Embarked', 'nLength_group', 'family_group', 'fare_group'], drop_first=False)
train.drop(['family_size', 'Name', 'Fare', 'name_length'], axis=1, inplace=True)
test.drop(['Name', 'family_size', "Fare", 'name_length'], axis=1, inplace=True)

# 年龄特征
# 使用随机森林回归量来预测缺失的年龄值，先确定有多少缺失值
from sklearn.ensemble import RandomForestRegressor
# 列重排
train = pd.concat([train[["Survived", "Age", "Sex", "SibSp", "Parch"]], train.loc[:, "is_alone":]], axis=1)
test = pd.concat([test[["Age", "Sex"]], test.loc[:, "SibSp":]], axis=1)


# 编写一个检测缺失值的函数，填充它
def completing_age(df):
    # 拿到除了存活之外的所有特征
    age_df = df.loc[:, "Age":]

    temp_train = age_df.loc[age_df.Age.notnull()]  # 有年龄值
    temp_test = age_df.loc[age_df.Age.isnull()]  # 年龄值缺失

    y = temp_train.Age.values  # 设置目标值在y轴
    x = temp_train.loc[:, "Sex":].values

    # 随机森林预测走起
    rfr = RandomForestRegressor(n_estimators=1500, n_jobs=-1)
    rfr.fit(x, y)

    predicted_age = rfr.predict(temp_test.loc[:, "Sex":])

    df.loc[df.Age.isnull(), "Age"] = predicted_age

    return df


# 查看缺失率
print("Train age missing value: " + str((train.Age.isnull().sum()/len(train))*100)+str("%"))
print("Test age missing value: " + str((test.Age.isnull().sum()/len(test))*100)+str("%"))
# 在train和test数据集中实现completion_age函数
train = completing_age(train)
test = completing_age(test)
print("Train age missing value: " + str((train.Age.isnull().sum()/len(train))*100)+str("%"))
print("Test age missing value: " + str((test.Age.isnull().sum()/len(test))*100)+str("%"))

# # 年龄柱状图
# plt.subplots(figsize=(22, 10),)
# sns.distplot(train.Age, bins=100, kde=True, rug=False, norm_hist=False)
# plt.show()

# # 年龄分组来创建新特性
# def age_group_fun(age):
#     a = ''
#     if age <= 1:
#         a = 'infant'
#     elif age <= 4:
#         a = 'toddler'
#     elif age <= 13:
#         a = 'child'
#     elif age <= 18:
#         a = 'teenager'
#     elif age <= 35:
#         a = 'Young_Adult'
#     elif age <= 45:
#         a = 'adult'
#     elif age <= 55:
#         a = 'middle_aged'
#     elif age <= 65:
#         a = 'senior_citizen'
#     else:
#         a = 'old'
#     return a


# train['age_group'] = train['Age'].map(age_group_fun)
# test['age_group'] = test['Age'].map(age_group_fun)

# # 创建age_group特征的虚拟变量
# train = pd.get_dummies(train, columns=['age_group'], drop_first=True)
# test = pd.get_dummies(test, columns=['age_group'], drop_first=True)

"""train.drop('Age', axis=1, inplace=True)
test.drop('Age', axis=1, inplace=True)"""

##########################################
#                建模开始                  #
##########################################
# 分离自变量因变量
X = train.drop(['Survived'], axis=1)
y = train["Survived"]

# 拆分训练数据
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=.33, random_state=0)

# 特征缩放
# print(train.sample())

# 准备工作
# headers = train_x.columns
# print(train_x.head())

# 特征缩放开始
# 使用标准转换器
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

# 转换x
train_x = sc.fit_transform(train_x)
# 转换"test_x"变量
test_x = sc.transform(test_x)
# 转换测试集
test = sc.transform(test)

# # 转换之后
# print(pd.DataFrame(train_x, columns=headers).head())