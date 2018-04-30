#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2018/3/8 12:59
# @Author  : xingyuezhiji
# @Email   : zhong180@126.com
# @File    : main.py
# @Software: PyCharm Community Edition
#coding=utf-8
import datetime
from sklearn.metrics import accuracy_score,recall_score,f1_score,precision_score,roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, \
    GradientBoostingClassifier,ExtraTreesClassifier,VotingClassifier
from mlxtend.classifier import StackingClassifier,EnsembleVoteClassifier
from sklearn import cross_validation
from sklearn.cross_validation import KFold
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
import matplotlib.pyplot as plt
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import ShuffleSplit
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


data1 = pd.read_csv("../data/f_train_20180204.csv",encoding='gb2312',)
data2 = pd.read_csv("../data/f_test_a_20180204.csv",encoding='gb2312')
data3 = pd.read_csv("../data/f_answer_a_20180306.csv",encoding='gb2312',names=['label'])

mydata = pd.concat([data2,data3],axis=1,)
mydata = pd.concat([data1,mydata])
mydata = pd.DataFrame(mydata)
mydata.to_csv("../data/train_add.csv",index = False)

print(mydata.columns)
print(mydata.isnull().sum()/len(mydata))
#特征处理
def change(name):
    try:
        df = pd.read_csv(name,encoding='gb2312')
    except:
        df = pd.read_csv(name)
    index = ['孕前BMI', 'SNP34','SNP46','SNP39','SNP42','SNP40',]
    index1 = ['SNP34','SNP20',]
    index2 = ['SNP34','SNP4','SNP47']
    index3 = ['SNP34', 'SNP7',]

    for i in index:
        # index.remove(i)
        for j in index:
            if i == j:
                continue
            else:
                df['{}/{}'.format(i,j)] = df[i]/df[j]

    for i in index1:
        index1.remove(i)
        for j in index1:
            if i == j:
                continue
            else:
                df['{}+{}'.format(i,j)] = df[i]+df[j]

    for i in index2:
        index2.remove(i)
        for j in index2:
            if i == j:
                continue
            else:
                df['{}*{}'.format(i,j)] = df[i]*df[j]
    for i in index3:
        index3.remove(i)
        for j in index3:
            if i == j:
                continue
            else:
                df['{}-{}'.format(i,j)] = df[i]-df[j]
    return df


df1 = change('../data/train_add.csv')
print(df1.shape)
df1.to_csv('../data/train_change.csv', index=False)
df2 = change('../data/f_test_a_20180204.csv')
df2.to_csv('../data/test_change.csv', index=False)
df3 = change('../data/f_test_b_20180305.csv')

df3.to_csv('../data/test_change_B.csv', index=False)

# 分类器
clf1 = lgb.LGBMClassifier(n_estimators=70,max_depth=6,num_leaves=64,colsample_bytree=0.3)
clf2 = xgb.XGBClassifier(n_estimators=70,max_depth=6,colsample_bytree=0.3)
ada = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(criterion="entropy",
                 splitter="best",max_depth=1,min_samples_split=2,min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,max_features=None,random_state=None,
                 max_leaf_nodes=None,min_impurity_split=1e-7,class_weight=None,presort=False),
                 n_estimators=60,learning_rate=0.1,
                 algorithm='SAMME.R',random_state=410)
gb =GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=70,
                               random_state=410, max_features=None, verbose=0,)
et = ExtraTreesClassifier(n_estimators=100,random_state=410)

#分类器集合
classifiers = [
    # clf1,clf2,gb,et,ada
    EnsembleVoteClassifier(clfs=[clf1,clf2,gb,et,ada],voting='soft',weights=[1,1,1,2,2]),
    # VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', gb)],voting='soft'),
    # StackingClassifier(classifiers=[clf1, clf2, gb],
    #                       meta_classifier=gb),
                ]


train = pd.read_csv('../data/train_change.csv')
test = pd.read_csv('../data/test_change.csv')
test_B = pd.read_csv('../data/test_change_B.csv')

# 均值填充
train.fillna(train.mean(axis=0), inplace=True)
test.fillna(test.mean(axis=0), inplace=True)
test_B.fillna(test.mean(axis=0), inplace=True)
print(train.columns)

train.drop(['id','ACEID',], axis=1, inplace=True)
test.drop(['id','ACEID',], axis=1, inplace=True)
test_B.drop(['id','ACEID',], axis=1, inplace=True)

# 特征与label划分
predictors = [f for f in train.columns if f not in ['label']]
x = train[predictors]
y = train['label']

#划分训练集和测试集
cv = ShuffleSplit(n_splits=5, test_size=0.16, random_state=888)
# kf = KFold(train.shape[0],n_folds=5,random_state=1)


accu = []
rec = []
auc = []
f1 = []
f1_t =[]
prec = []
i=0
#五折训练，取平均
for tr, te in cv.split(x): #tr：五折训练的训练集标签；tr：
    i += 1
    ensemble = []
    recall_ensemble =[]
    train_t_ensemble = []

    print('第'+str(i)+'次训练:')
    for clf in classifiers:
        train_predictors = train[predictors].iloc[tr] # 将predictors作为测试特征
        train_target = train['label'].iloc[tr]
        clf.fit(train_predictors, train_target)

        name = clf.__class__.__name__
        print("=" * 30)
        print(name)

        print('****Results****')

        train_predictions = clf.predict(train[predictors].iloc[te]) #预测标签
        train_proba = clf.predict_proba(train[predictors].iloc[te]) #预测概率
        train_t = np.round(1/(1+np.exp(train_proba[:,0]-train_proba[:,1]))+0.002)
        #1/(e^(a-b)+1),利用预测概率以及sigmoid函数将预测标签微调(目的:增加正例个数)

        ensemble.append(train_predictions)
        train_t_ensemble.append(train_t)

        acc = accuracy_score(train['label'].iloc[te], train_predictions)

        F1 = f1_score(train['label'].iloc[te], train_predictions)
        F1_t = f1_score(train['label'].iloc[te], train_t)
        recall = recall_score(train['label'].iloc[te], train_predictions)


        # 输出每一次训练的Accuracy，F1值以及Recall值
        print("Accuracy: {:.4%}".format(acc))
        print("F1: {:.4%}".format(F1))
        print("Recall: {:.4%}".format(recall))



    ensemble = (np.sum(ensemble,axis=0))//int((len(classifiers)+1)/2)
    train_t_ensemble = (np.sum(train_t_ensemble,axis=0))//int((len(classifiers)+1)/2)
    # recall_ensemble = (np.sum(recall_ensemble, axis=0)) // int((len(classifiers) + 1) / 2)
    # print(ensemble)

    print(accuracy_score(train['label'].iloc[te], ensemble))
    accu.append(accuracy_score(train['label'].iloc[te], ensemble))
    rec.append(recall_score(train['label'].iloc[te], ensemble))
    auc.append(roc_auc_score(train['label'].iloc[te], ensemble))
    f1.append(f1_score(train['label'].iloc[te], ensemble))
    f1_t.append(f1_score(train['label'].iloc[te], train_t_ensemble))
    prec.append(precision_score(train['label'].iloc[te], ensemble))


test_ensemble = []
prob_ensemble = []

# 全数据集预测
for clf in classifiers:
    clf.fit(train[predictors],train['label'])
    test_prediction = clf.predict(test_B[predictors])
    test_ensemble.append(test_prediction)
    prob = clf.predict_proba(test_B[predictors])

    # prob_ensemble.append(prob)


    # importance = clf.feature_importances_
    #
    # feature_importance = clf.feature_importances_
    # feature_importance = 100.0 * (feature_importance / feature_importance.max())
    #
    # fi_threshold = 35
    # important_idx = np.where(feature_importance > fi_threshold)[0]
    # important_features = train.columns.values[1::][important_idx]
    # print("\n", important_features.shape[0], "Important features(>", fi_threshold, "% of max importance)...\n")
    # # important_features
    # sorted_idx = np.argsort(feature_importance[important_idx])[::-1]
    # # get the figure about important features
    # pos = np.arange(sorted_idx.shape[0]) + 0.5
    # plt.subplot(1, 2, 2)
    # plt.title('Feature Importance')
    # plt.barh(pos, feature_importance[important_idx][sorted_idx[::-1]],
    #          color='r', align='center')
    # plt.yticks(pos, important_features[sorted_idx[::-1]], fontproperties=font)
    # plt.xlabel('Relative Importance')
    # plt.draw()
    # plt.show()

# test_ensemble = (np.sum(test_ensemble,axis=0))//int((len(classifiers)+1)/2)
# prob_ensemble = (np.sum(prob_ensemble,axis=0))//int((len(classifiers)+1)/2)

test_t = np.round(1/(1+np.exp(prob[:,0]-prob[:,1]))+0.002)
# print('prob:',np.round(1/(1+np.exp(prob[:,0]-prob[:,1]))))
print('Accu_mean',np.mean(accu))

# print('prob:{:.4%}'.format(np.mean(prob_ensemble)))

print("Recall_mean: {:.4%}".format(np.mean(rec)))
print("Precision_mean: {:.4%}".format(np.mean(prec)))
print("Auc_score: {:.4%}".format(np.mean(auc)))
print("F1_score: {:.4%}".format(np.mean(f1_t)))
# print(test_ensemble)
# print(np.sum(test_ensemble))
print(np.sum(test_t)) #输出正例的个数

submission = pd.DataFrame({'sub': test_t})

# submission.to_csv(r'../submit/submit_{}.csv'.format(datetime.datetime.now().
#     strftime('%Y%m%d_%H%M%S')), header=None,index=False, float_format='%.4f')