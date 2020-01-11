#-*- coding: utf-8 -*-

# 主要用来分析数据集合与我们遇到的问题

from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.ensemble import GradientBoostingRegressor
from mlxtend.regressor import StackingCVRegressor
from sklearn.preprocessing import RobustScaler
from scipy.stats import boxcox_normmax, zscore
from multiprocessing import cpu_count
from lightgbm import LGBMRegressor
from scipy.special import boxcox1p
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.svm import SVR
import pandas as pd
import numpy as np

#中文支持matplotlib
plt.rcParams['font.sans-serif'] = ['SimHei']  # 步骤一（替换sans-serif字体）
plt.rcParams['axes.unicode_minus'] = False  # 步骤二（解决坐标轴负数的负号显示问题）

kf = KFold(n_splits=5, random_state=0, shuffle=True)
rmse = lambda y, y_pred: np.sqrt(mean_squared_error(y, y_pred))
scorer = make_scorer(rmse, greater_is_better=False)

# 加载数据
train = pd.read_csv("../data/train.csv", index_col=0)
test = pd.read_csv("../data/test.csv", index_col=0)
sample = pd.read_csv("../data/sample_submission.csv")

# 分析训练数据
def anaylysis_data(train):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.grid()
    ax.scatter(train["GrLivArea"], train["SalePrice"], c="#3f72af", zorder=3, alpha=0.9)
    ax.axvline(4500, c="#112d4e", ls="--", zorder=2)
    ax.set_xlabel("GrLivArea", labelpad=10)
    ax.set_ylabel(u"房价 ($)", labelpad=10)
    ax.set_title('房价与GrLiveArea的关系', fontsize=12, color='r')
    plt.savefig("房价与GrLiveArea的关系.jpg")
    plt.show()

# 训练数据处理
def handle_traind_data(train):
    train = train[train["GrLivArea"] < 4500]
    X = pd.concat([train.drop("SalePrice", axis=1), test])

    #目标值
    y_train = np.log(train["SalePrice"])
    return X, y_train

# 缺失值处理
# 1、分析缺失值：statistics_misssing_values
def statistics_misssing_values(X):
    nans = X.isna().sum().sort_values(ascending=False)
    nans = nans[nans > 0]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.grid()
    ax.bar(nans.index, nans.values, zorder=2, color="#3f72af")
    ax.set_ylabel("No. of missing values", labelpad=10)
    ax.set_xlim(-0.6, len(nans) - 0.4)
    ax.xaxis.set_tick_params(rotation=90)
    ax.set_title('数据缺失统计', fontsize=12, color='r')
    plt.savefig("数据缺失统计.jpg")
    plt.show()

# 2、处理缺失值： handle_missing_values
def handle_missing_values(X):
    cols = ["PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu", "GarageCond", "GarageQual", "GarageFinish",
            "GarageType", "BsmtCond", "BsmtExposure", "BsmtQual", "BsmtFinType2", "BsmtFinType1"]
    X[cols] = X[cols].fillna("None")
    cols = ["GarageYrBlt", "MasVnrArea", "BsmtHalfBath", "BsmtFullBath", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF",
            "TotalBsmtSF", "GarageCars"]
    X[cols] = X[cols].fillna(0)
    cols = ["MasVnrType", "MSZoning", "Utilities", "Exterior1st", "Exterior2nd", "SaleType", "Electrical",
            "KitchenQual", "Functional"]
    X[cols] = X.groupby("Neighborhood")[cols].transform(lambda x: x.fillna(x.mode()[0]))
    cols = ["GarageArea", "LotFrontage"]
    X[cols] = X.groupby("Neighborhood")[cols].transform(lambda x: x.fillna(x.median()))
    return X


# 特征工程：  features_engineer
def features_engineer(X):
    X["TotalSF"] = X["GrLivArea"] + X["TotalBsmtSF"]
    X["TotalPorchSF"] = X["OpenPorchSF"] + X["EnclosedPorch"] + X["3SsnPorch"] + X["ScreenPorch"]
    X["TotalBath"] = X["FullBath"] + X["BsmtFullBath"] + 0.5 * (X["BsmtHalfBath"] + X["HalfBath"])

    cols = ["MSSubClass", "YrSold"]
    X[cols] = X[cols].astype("category")

    X["SinMoSold"] = np.sin(2 * np.pi * X["MoSold"] / 12)
    X["CosMoSold"] = np.cos(2 * np.pi * X["MoSold"] / 12)
    X = X.drop("MoSold", axis=1)

    skew = X.skew(numeric_only=True).abs()
    cols = skew[skew > 1].index
    for col in cols:
        X[col] = boxcox1p(X[col], boxcox_normmax(X[col] + 1))

    cols = X.select_dtypes(np.number).columns
    X[cols] = RobustScaler().fit_transform(X[cols])

    X = pd.get_dummies(X)

    X_train = X.loc[train.index]
    X_test = X.loc[test.index]
    return  X_train, X_test

# 去除异常值： Remove outliers from training data
def remove_outliers_from_training_data(X_train, y_train):
    residuals = y_train - LinearRegression().fit(X_train, y_train).predict(X_train)
    outliers = residuals[np.abs(zscore(residuals)) > 3].index

    X_train = X_train.drop(outliers)
    y_train = y_train.drop(outliers)
    return X_train, y_train



# 交叉验证
def random_search(X_train, y_train, model, grid, n_iter=100):
    n_jobs = max(cpu_count() - 2, 1)
    search = RandomizedSearchCV(model, grid, n_iter, scorer, n_jobs=n_jobs, cv=kf, random_state=0, verbose=True)
    return search.fit(X_train, y_train)

# Ridge 方法
def ridge_meathod(X_train, y_train):
    ridge_search = random_search(X_train, y_train, Ridge(), {"alpha": np.logspace(-1, 2, 500)})
    print(ridge_search)
    return ridge_search

if __name__=="__main__":
    # 分析训练数据
    anaylysis_data(train)
    # 处理训练数据
    X, y_train = handle_traind_data(train)
    #print(X, y_train)
    # 缺失值分析
    statistics_misssing_values(X)
    # 处理缺失值
    X = handle_missing_values(X)
    #print(X)
    # 特征工程
    X_train, X_test = features_engineer(X)
    #print(X_train, X_test)
    # 去除异常值
    X_train, y_train = remove_outliers_from_training_data(X_train, y_train)
    # 用ridge方法
    #ridge_result = ridge_meathod(X_train, y_train)


