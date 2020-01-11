#-*- coding: utf-8 -*-

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

#读取数据
train = pd.read_csv("../data/train.csv", index_col=0)
test = pd.read_csv("../data/test.csv", index_col=0)
sample = pd.read_csv("../data/sample_submission.csv")

#研究训练数据
fig, ax = plt.subplots(figsize=(10, 6))
ax.grid()
ax.scatter(train["GrLivArea"], train["SalePrice"], c="#3f72af", zorder=3, alpha=0.9)
ax.axvline(4500, c="#112d4e", ls="--", zorder=2)
ax.set_xlabel("Ground living area (sq. ft)", labelpad=10)
ax.set_ylabel("Sale price ($)", labelpad=10)
plt.savefig("训练数据.jpg")
plt.show()

# 训练数据处理
train = train[train["GrLivArea"] < 4500]
X = pd.concat([train.drop("SalePrice", axis=1), test])

#目标值
y_train = np.log(train["SalePrice"])

# 处理缺失值
nans = X.isna().sum().sort_values(ascending=False)
nans = nans[nans > 0]
fig, ax = plt.subplots(figsize=(10, 6))
ax.grid()
ax.bar(nans.index, nans.values, zorder=2, color="#3f72af")
ax.set_ylabel("No. of missing values", labelpad=10)
ax.set_xlim(-0.6, len(nans) - 0.4)
ax.xaxis.set_tick_params(rotation=90)
plt.savefig("处理缺失值.jpg")
plt.show()

#补充缺失值
cols = ["PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu", "GarageCond", "GarageQual", "GarageFinish", "GarageType", "BsmtCond", "BsmtExposure", "BsmtQual", "BsmtFinType2", "BsmtFinType1"]
X[cols] = X[cols].fillna("None")
cols = ["GarageYrBlt", "MasVnrArea", "BsmtHalfBath", "BsmtFullBath", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "GarageCars"]
X[cols] = X[cols].fillna(0)
cols = ["MasVnrType", "MSZoning", "Utilities", "Exterior1st", "Exterior2nd", "SaleType", "Electrical", "KitchenQual", "Functional"]
X[cols] = X.groupby("Neighborhood")[cols].transform(lambda x: x.fillna(x.mode()[0]))
cols = ["GarageArea", "LotFrontage"]
X[cols] = X.groupby("Neighborhood")[cols].transform(lambda x: x.fillna(x.median()))

# 建立特征工程
X["TotalSF"] = X["GrLivArea"] + X["TotalBsmtSF"]
X["TotalPorchSF"] = X["OpenPorchSF"] + X["EnclosedPorch"] + X["3SsnPorch"] + X["ScreenPorch"]
X["TotalBath"] = X["FullBath"] + X["BsmtFullBath"] + 0.5 * (X["BsmtHalfBath"] + X["HalfBath"])

cols = ["MSSubClass", "YrSold"]
X[cols] = X[cols].astype("category")

# 特征处理
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

residuals = y_train - LinearRegression().fit(X_train, y_train).predict(X_train)
outliers = residuals[np.abs(zscore(residuals)) > 3].index

X_train = X_train.drop(outliers)
y_train = y_train.drop(outliers)

kf = KFold(n_splits=5, random_state=0, shuffle=True)
rmse = lambda y, y_pred: np.sqrt(mean_squared_error(y, y_pred))
scorer = make_scorer(rmse, greater_is_better=False)

def random_search(model, grid, n_iter=100):
    n_jobs = max(cpu_count() - 2, 1)
    search = RandomizedSearchCV(model, grid, n_iter, scorer, n_jobs=n_jobs, cv=kf, random_state=0, verbose=True)
    return search.fit(X_train, y_train)

# ridge_search = random_search(Ridge(), {"alpha": np.logspace(-1, 2, 500)})
# lasso_search = random_search(Lasso(), {"alpha": np.logspace(-5, -1, 500)})
# svr_search = random_search(SVR(), {"C": np.arange(1, 100), "gamma": np.linspace(0.00001, 0.001, 50), "epsilon": np.linspace(0.01, 0.1, 50)})
# lgbm_search = random_search(LGBMRegressor(n_estimators=2000, max_depth=3), {"colsample_bytree": np.linspace(0.2, 0.7, 6), "learning_rate": np.logspace(-3, -1, 100)})
# gbm_search = random_search(GradientBoostingRegressor(n_estimators=2000, max_depth=3), {"max_features": np.linspace(0.2, 0.7, 6), "learning_rate": np.logspace(-3, -1, 100)})
#
# models = [search.best_estimator_ for search in [ridge_search, lasso_search, svr_search, lgbm_search, gbm_search]]
# stack_search = random_search(StackingCVRegressor(models, Ridge(), cv=kf), {"meta_regressor__alpha": np.logspace(-3, -2, 500)}, n_iter=20)
# models.append(stack_search.best_estimator_)
#
# preds = [model.predict(X_test) for model in models]
# preds.append(np.log(pd.read_csv("blend-linear-regressors.csv")["SalePrice"]))
#
# preds = np.average(preds, axis=0, weights=[0.1] * 5 + [0.25] * 2)
# submission = pd.DataFrame({"Id": sample["Id"], "SalePrice": np.exp(preds)})
# submission.to_csv("submission.csv", index=False)

# Ridge() 计算结果
def ridge_test():
    ridge_search = random_search(Ridge(), {"alpha": np.logspace(-1, 2, 500)})
    models = [search.best_estimator_ for search in [ridge_search]]
    stack_search = random_search(StackingCVRegressor(models, Ridge(), cv=kf), {"meta_regressor__alpha": np.logspace(-3, -2, 500)}, n_iter=20)
    models.append(stack_search.best_estimator_)

    preds = [model.predict(X_test) for model in models]
    ridge_result = preds[0]

    result = pd.read_csv("../data/submission.csv", index_col=0)

    ridge_value_result = np.exp(ridge_result)
    print("---")
    print(result["SalePrice"])

    matplotlib.use('TkAgg')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.grid()
    ax.scatter(ridge_value_result, result["SalePrice"], c="#3f72af", zorder=3, alpha=0.9)
    ax.axvline(4500, c="#112d4e", ls="--", zorder=2)
    ax.set_xlabel(u"Ridge计算结果", labelpad=10)
    ax.set_ylabel("最终结果", labelpad=10)
    plt.savefig("Ridge计算结果.jpg")
    plt.show()
    #


# Lasso计算结果
def lasso_test():
    lasso_search = random_search(Lasso(), {"alpha": np.logspace(-5, -1, 500)})
    models = [search.best_estimator_ for search in [lasso_search]]
    stack_search = random_search(StackingCVRegressor(models, Ridge(), cv=kf),
                                 {"meta_regressor__alpha": np.logspace(-3, -2, 500)}, n_iter=20)
    models.append(stack_search.best_estimator_)

    preds = [model.predict(X_test) for model in models]
    lasso_search = preds[0]

    result = pd.read_csv("../data/submission.csv", index_col=0)

    lasso_value_result = np.exp(lasso_search)
    print("---")
    print(result["SalePrice"])

    matplotlib.use('TkAgg')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.grid()
    ax.scatter(lasso_value_result, result["SalePrice"], c="#3f72af", zorder=3, alpha=0.9)
    ax.axvline(4500, c="#112d4e", ls="--", zorder=2)
    ax.set_xlabel(u"Lasso计算结果", labelpad=10)
    ax.set_ylabel("对照结果", labelpad=10)
    plt.savefig("Lasso计算结果.jpg")
    plt.show()

# SVR计算
def svr_test():
    svr_search = random_search(SVR(), {"C": np.arange(1, 100), "gamma": np.linspace(0.00001, 0.001, 50), "epsilon": np.linspace(0.01, 0.1, 50)})
    models = [search.best_estimator_ for search in [svr_search]]
    stack_search = random_search(StackingCVRegressor(models, Ridge(), cv=kf),
                                 {"meta_regressor__alpha": np.logspace(-3, -2, 500)}, n_iter=20)
    models.append(stack_search.best_estimator_)

    preds = [model.predict(X_test) for model in models]
    svr_search = preds[0]

    result = pd.read_csv("../data/submission.csv", index_col=0)

    svr_value_result = np.exp(svr_search)
    print("---")
    print(result["SalePrice"])

    matplotlib.use('TkAgg')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.grid()
    ax.scatter(svr_value_result, result["SalePrice"], c="#3f72af", zorder=3, alpha=0.9)
    ax.axvline(4500, c="#112d4e", ls="--", zorder=2)
    ax.set_xlabel(u"svr计算结果", labelpad=10)
    ax.set_ylabel("对照结果", labelpad=10)
    plt.savefig("SVR计算结果.jpg")
    plt.show()


# LGBM 方法
def lgbm_test():
    lgbm_search = random_search(LGBMRegressor(n_estimators=2000, max_depth=3), {"colsample_bytree": np.linspace(0.2, 0.7, 6), "learning_rate": np.logspace(-3, -1, 100)})
    models = [search.best_estimator_ for search in [lgbm_search]]
    stack_search = random_search(StackingCVRegressor(models, Ridge(), cv=kf),
                                 {"meta_regressor__alpha": np.logspace(-3, -2, 500)}, n_iter=20)
    models.append(stack_search.best_estimator_)

    preds = [model.predict(X_test) for model in models]
    lgbm_search = preds[0]

    result = pd.read_csv("../data/submission.csv", index_col=0)

    lgbm_value_result = np.exp(lgbm_search)
    print("---")
    print(result["SalePrice"])

    matplotlib.use('TkAgg')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.grid()
    ax.scatter(lgbm_value_result, result["SalePrice"], c="#3f72af", zorder=3, alpha=0.9)
    ax.axvline(4500, c="#112d4e", ls="--", zorder=2)
    ax.set_xlabel(u"LGBM计算结果", labelpad=10)
    ax.set_ylabel("对照结果", labelpad=10)
    plt.savefig("LGBM计算结果.jpg")
    plt.show()


# # 采用GradientBoosting 方法: 计算机跑不过，废弃不用
# def gbm_test():
#     gbm_search = random_search(GradientBoostingRegressor(n_estimators=2000, max_depth=3), {"max_features": np.linspace(0.2, 0.7, 6), "learning_rate": np.logspace(-3, -1, 100)})
#     models = [search.best_estimator_ for search in [gbm_search]]
#     stack_search = random_search(StackingCVRegressor(models, Ridge(), cv=kf),
#                                  {"meta_regressor__alpha": np.logspace(-3, -2, 500)}, n_iter=20)
#     models.append(stack_search.best_estimator_)
#
#     preds = [model.predict(X_test) for model in models]
#     gbm_search = preds[0]
#
#     result = pd.read_csv("../data/submission.csv", index_col=0)
#
#     gbm_value_result = np.exp(gbm_search)
#     print("---")
#     print(result["SalePrice"])
#
#     matplotlib.use('TkAgg')
#     fig, ax = plt.subplots(figsize=(10, 6))
#     ax.grid()
#     ax.scatter(gbm_value_result, result["SalePrice"], c="#3f72af", zorder=3, alpha=0.9)
#     ax.axvline(4500, c="#112d4e", ls="--", zorder=2)
#     ax.set_xlabel(u"GBM计算结果", labelpad=10)
#     ax.set_ylabel("对照结果", labelpad=10)
#     plt.savefig("GBM计算结果.jpg")
#     plt.show()


def ensemble_learning():
    ridge_search = random_search(Ridge(), {"alpha": np.logspace(-1, 2, 500)})
    lasso_search = random_search(Lasso(), {"alpha": np.logspace(-5, -1, 500)})
    svr_search = random_search(SVR(), {"C": np.arange(1, 100), "gamma": np.linspace(0.00001, 0.001, 50), "epsilon": np.linspace(0.01, 0.1, 50)})
    lgbm_search = random_search(LGBMRegressor(n_estimators=2000, max_depth=3), {"colsample_bytree": np.linspace(0.2, 0.7, 6), "learning_rate": np.logspace(-3, -1, 100)})

    models = [search.best_estimator_ for search in [ridge_search, lasso_search, svr_search, lgbm_search]]
    stack_search = random_search(StackingCVRegressor(models, Ridge(), cv=kf), {"meta_regressor__alpha": np.logspace(-3, -2, 500)}, n_iter=20)
    models.append(stack_search.best_estimator_)

    preds = [model.predict(X_test) for model in models]

    preds = np.average(preds, axis=0)

    result = pd.read_csv("../data/submission.csv", index_col=0)

    value_result = np.exp(preds)
    matplotlib.use('TkAgg')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.grid()
    ax.scatter(value_result, result["SalePrice"], c="#3f72af", zorder=3, alpha=0.9)
    ax.axvline(4500, c="#112d4e", ls="--", zorder=2)
    ax.set_xlabel(u"集成学习计算结果", labelpad=10)
    ax.set_ylabel("对照结果", labelpad=10)
    plt.savefig("集成学习计算结果.jpg")
    plt.show()

if __name__=="__main__":
    ensemble_learning()

