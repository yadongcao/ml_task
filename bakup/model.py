#-*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# 先数据处理: 一共81一个数据值

from ml_task.bakup.data_dict import *


class Model(object):
    def __init__(self):
        self.ids=None
        pass

    def load_train_data(self):
        path="./data/train.csv"
        train_df = pd.read_csv(path)
        return train_df

    def load_test_data(self):
        path="./data/test.csv"
        train_df = pd.read_csv(path)
        return train_df

    def handle_data_source(self, data_source_df):
        data_source_df["MSZoning"] = data_source_df["MSZoning"].replace(MSZoning_dict)

        data_source_df["Street"] = data_source_df["Street"].replace(Street_dict)

        data_source_df["Alley"] = data_source_df["Alley"].replace(Alley_dict)

        data_source_df["LotShape"] = data_source_df["LotShape"].replace(LotShape_dict)

        data_source_df["LandContour"] = data_source_df["LandContour"].replace(LandContour_dict)

        data_source_df["Utilities"] = data_source_df["Utilities"].replace(Utilities_dict)

        data_source_df["LotConfig"] = data_source_df["LotConfig"].replace(LotConfig_dict)

        data_source_df["LandSlope"] = data_source_df["LandSlope"].replace(LandSlope_dict)

        data_source_df["Neighborhood"] = data_source_df["Neighborhood"].replace(Neighborhood_dict)

        data_source_df["Condition1"] = data_source_df["Condition1"].replace(Condition1_dict)

        data_source_df["Condition2"] = data_source_df["Condition2"].replace(Condition2_dict)

        data_source_df["BldgType"] = data_source_df["BldgType"].replace(BldgType_dict)

        data_source_df["HouseStyle"] = data_source_df["HouseStyle"].replace(HouseStyle_dict)

        data_source_df["RoofStyle"] = data_source_df["RoofStyle"].replace(RoofStyle_dict)

        data_source_df["RoofMatl"] = data_source_df["RoofMatl"].replace(RoofMatl_dict)

        data_source_df["Exterior1st"] = data_source_df["Exterior1st"].replace(Exterior1st_dict)

        data_source_df["Exterior2nd"] = data_source_df["Exterior2nd"].replace(Exterior2nd_dict)

        data_source_df["MasVnrType"] = data_source_df["MasVnrType"].replace(MasVnrType_dict)

        data_source_df["ExterQual"] = data_source_df["ExterQual"].replace(ExterQual_dict)

        data_source_df["ExterCond"] = data_source_df["ExterCond"].replace(ExterCond_dict)

        data_source_df["Foundation"] = data_source_df["Foundation"].replace(Foundation_dict)

        data_source_df["BsmtQual"] = data_source_df["BsmtQual"].replace(BsmtQual_dict)

        data_source_df["BsmtCond"] = data_source_df["BsmtCond"].replace(BsmtCond_dict)

        # print(data_source_df["BsmtCond"])
        # data_source_df["BsmtCond"] = data_source_df["BsmtCond"].replace(BsmtCond_dict)

        data_source_df["BsmtExposure"] = data_source_df["BsmtExposure"].replace(BsmtExposure_dict)

        data_source_df["BsmtFinType1"] = data_source_df["BsmtFinType1"].replace(BsmtFinType1_dict)

        data_source_df["BsmtFinType2"] = data_source_df["BsmtFinType2"].replace(BsmtFinType2_dict)

        data_source_df["Heating"] = data_source_df["Heating"].replace(Heating_dict)

        data_source_df["HeatingQC"] = data_source_df["HeatingQC"].replace(HeatingQC_dict)

        data_source_df["CentralAir"] = data_source_df["CentralAir"].replace(CentralAir_dict)

        data_source_df["Electrical"] = data_source_df["Electrical"].replace(Electrical_dict)

        data_source_df["KitchenQual"] = data_source_df["KitchenQual"].replace(KitchenQual_dict)

        data_source_df["Functional"] = data_source_df["Functional"].replace(Functional_dict)

        data_source_df["FireplaceQu"] = data_source_df["FireplaceQu"].replace(FireplaceQu_dict)

        data_source_df["GarageType"] = data_source_df["GarageType"].replace(GarageType_dict)

        data_source_df["GarageFinish"] = data_source_df["GarageFinish"].replace(GarageFinish_dict)

        data_source_df["GarageQual"] = data_source_df["GarageQual"].replace(GarageQual_dict)

        data_source_df["GarageCond"] = data_source_df["GarageCond"].replace(GarageCond_dict)

        data_source_df["PavedDrive"] = data_source_df["PavedDrive"].replace(PavedDrive_dict)

        data_source_df["PoolQC"] = data_source_df["PoolQC"].replace(PoolQC_dict)

        data_source_df["Fence"] = data_source_df["Fence"].replace(Fence_dict)

        data_source_df["MiscFeature"] = data_source_df["MiscFeature"].replace(MiscFeature_dict)

        data_source_df["SaleType"] = data_source_df["SaleType"].replace(SaleType_dict)

        data_source_df["SaleCondition"] = data_source_df["SaleCondition"].replace(SaleCondition_dict)

        return data_source_df



    # 数据处理，在dataframe 中处理
    def handle_source_train_data(self):
        train_df = self.load_train_data()
        handled_tradin_df = self.handle_data_source(train_df)
        #print(handled_tradin_df)

        #print(train_df.head(10))
        #handled_tradin_df.to_csv(r".\data\handled_train.csv", index=False)
        return handled_tradin_df

    # 数据处理，在dataframe 中处理
    def handle_source_test_data(self):
        train_df = self.load_test_data()
        self.ids = train_df["Id"]
        handled_tradin_df = self.handle_data_source(train_df)
        # print(handled_tradin_df)

        # print(train_df.head(10))
        # handled_tradin_df.to_csv(r".\data\handled_train.csv", index=False)
        return handled_tradin_df

    # 对数据进行特征规范化
    def normalize_feature(self, dataframe, f_min=0, f_max=1.0):
        #dataframe = self.handle_source_train_data()
        columns_name_list = dataframe.columns.tolist()
        need_delete_columns = ["Id", "SalePrice"]
        normalize_columns_list = list(set(columns_name_list) - set(need_delete_columns))
        for i in range(len(normalize_columns_list)):
            temp_column_name = normalize_columns_list[i]
            d_min, d_max = np.min(dataframe[temp_column_name]), np.max(dataframe[temp_column_name])
            factor = (f_max - f_min) / (d_max - d_min)
            normalized = f_min + (dataframe[temp_column_name] - d_min) * factor
            dataframe[temp_column_name] = normalized
        return dataframe

    # 训练数据特征规范化
    def train_data_normalize_feature(self):
        dataframe = self.handle_source_train_data()
        normalize_feature_data = self.normalize_feature(dataframe)
        normalize_feature_data.to_csv("./data/normalize_train.csv", index=False)
        #print(normalize_feature_data)
        return normalize_feature_data

    # 测试数据特征规范化
    def test_data_normalize_feature(self):
        dataframe = self.handle_source_test_data()
        normalize_feature_data = self.normalize_feature(dataframe)
        #normalize_feature_data.to_csv("./data/normalize_train.csv", index=False)
        # print(normalize_feature_data)
        return normalize_feature_data

    # 切分数据：回归矩阵与值向量
    def load_model_data(self):
        dataframe = self.train_data_normalize_feature()
        labelMat = dataframe["SalePrice"].tolist()
        need_delete_columns = ["Id", "SalePrice"]
        columns_name_list = dataframe.columns.tolist()
        data_columns_list = list(set(columns_name_list) - set(need_delete_columns))
        data_columns_list=sorted(data_columns_list)
        temp_dataframe = dataframe[data_columns_list]
        temp_dataframe.fillna(0, inplace=True)
        dataMat = temp_dataframe.values.tolist()
        return dataMat, labelMat

    # 测试数据
    def load_test_final_data(self):
        dataframe = self.test_data_normalize_feature()
        #ids = dataframe["id"]
        need_delete_columns = ["Id", "SalePrice"]
        columns_name_list = dataframe.columns.tolist()
        data_columns_list = list(set(columns_name_list) - set(need_delete_columns))
        data_columns_list = sorted(data_columns_list)
        temp_dataframe = dataframe[data_columns_list]
        temp_dataframe.fillna(0, inplace=True)
        dataMat = temp_dataframe.values.tolist()

        #print(dataMat)
        return dataMat


    # 标准回归函数
    def stand_regres(self, xArr, yArr):
        xMat = np.mat(xArr)
        yMat = np.mat(yArr).T
        xTx = xMat.T * xMat
        if np.linalg.det(xTx) == 0.0:
            print("This matrix is singular, cannot do inverse")
            return
        ws = xTx.I * (xMat.T * yMat)
        return ws

    # 标准回归测试
    def stand_reres_test(self):
        xArr, yArr = self.load_model_data()
        #print(type(xArr), xArr)
        ws = self.stand_regres(xArr, yArr)
        #print(yArr)

        xMat = np.mat(xArr)
        yMat = np.mat(yArr)
        yHat = xMat * ws
        print(yHat)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(xMat[:,1].flatten().A[0], yMat.T[:, 0].flatten().A[0])
        xCopy = xMat.copy()
        xCopy.sort(0)
        yHat=xCopy*ws
        #ax.plot(xCopy[:, 1], yHat)
        plt.show()

    # 采用局部加权线性回归
    def lwlr(self, testPoint, xArr, yArr, k=1.0):
        xMat = np.mat(xArr)
        yMat = np.mat(yArr).T
        m = np.shape(xMat)[0]
        weights = np.mat(np.eye((m)))
        for j in range(m):  # next 2 lines create weights matrix
            diffMat = testPoint - xMat[j, :]  #
            weights[j, j] = np.exp(diffMat * diffMat.T / (-2.0 * k ** 2))
        xTx = xMat.T * (weights * xMat)
        if np.linalg.det(xTx) == 0.0:
            print("This matrix is singular, cannot do inverse")
            return
        ws = xTx.I * (xMat.T * (weights * yMat))
        return testPoint * ws

    def lwlr_test(self, testArr, xArr,yArr,k=1.0):
        m = np.shape(testArr)[0]
        yHat = np.zeros(m)
        for i in range(m):
            yHat[i] = self.lwlr(testArr[i], xArr, yArr, k)
        return yHat

    # 局部加权线性回归测试
    def lwlr_reres_test(self):
        xArr, yArr = self.load_model_data()
        # print(type(xArr), xArr)
        yHat = self.lwlr_test(xArr, xArr, yArr, 0.7)
        # print(yArr)
        #print(ws)
        xMat = np.mat(xArr)
        srtInd = xMat[:, 1].argsort(0)
        xSort = xMat[srtInd][:, 0, :]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(xSort[:,1], yHat[srtInd])
        ax.scatter(xMat[:, 1].flatten().A[0], np.mat(yArr).T.flatten().A[0], s= 2, c='red')
        plt.show()

    # 采用领回归
    def ridge_regres(self, xMat, yMat, lam=0.2):
        xTx = xMat.T * xMat
        denom = xTx + np.eye(np.shape(xMat)[1]) * lam
        if np.linalg.det(denom) == 0.0:
            print("This matrix is singular, cannot do inverse")
            return
        ws = denom.I * (xMat.T * yMat)
        return ws

    def ridge_test(self, xArr, yArr):
        xMat = np.mat(xArr)
        yMat = np.mat(yArr).T
        yMean = np.mean(yMat, 0)
        yMat = yMat - yMean  # to eliminate X0 take mean off of Y
        # regularize X's
        xMeans = np.mean(xMat, 0)  # calc mean then subtract it off
        xVar = np.var(xMat, 0)  # calc variance of Xi then divide by it
        xMat = (xMat - xMeans) / xVar
        numTestPts = 30
        wMat = np.zeros((numTestPts, np.shape(xMat)[1]))
        for i in range(numTestPts):
            ws = self.ridge_regres(xMat, yMat, np.exp(i - 10))
            wMat[i, :] = ws.T
        return wMat

    # 领回归测试
    def ridge_test_result(self):
        abX, abY = self.load_model_data()
        abX = np.mat(abX)
        abY = np.mat(abY)
        ridge_weights = self.ridge_test(abX, abY)
        print(ridge_weights)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(ridge_weights)
        plt.show()

    def rssError(self, yArr, yHatArr):  # yArr and yHatArr both need to be arrays
        return ((yArr - yHatArr) ** 2).sum()

    # # 在岭回归中找最佳权重
    # def cross_validation(self, xArr, yArr, numVal=10):
    #     m = len(yArr)
    #     indexList = np.arange(m)
    #     errorMat = np.zeros((numVal, 30))  # create error mat 30columns numVal rows
    #     for i in range(numVal):
    #         trainX = []
    #         trainY = []
    #         testX = []
    #         testY = []
    #         np.random.shuffle(indexList)
    #         for j in range(m):  # create training set based on first 90% of values in indexList
    #             #print(np.array(xArr[indexList[j]]).tolist()[0], np.array(yArr[indexList[j]]).tolist()[0])
    #             if j < m * 0.9:
    #                 trainX.append(xArr[indexList[j]])
    #                 trainY.append(yArr[indexList[j]])
    #             else:
    #                 testX.append(xArr[indexList[j]])
    #                 testY.append(yArr[indexList[j]])
    #         #print(trainX, trainY)
    #         wMat = self.ridge_test(trainX, trainY)  # get 30 weight vectors from ridge
    #         #print(wMat)
    #         for k in range(30):  # loop over all of the ridge estimates
    #             matTestX = np.mat(testX)
    #             matTrainX = np.mat(trainX)
    #             meanTrain = np.mean(matTrainX, 0)
    #             varTrain = np.var(matTrainX, 0)
    #             #print(matTestX, meanTrain, varTrain)
    #             matTestX = (matTestX - meanTrain) / varTrain  # regularize test with training params
    #             yEst = matTestX * np.mat(wMat[k, :]).T + np.mean(trainY)  # test ridge results and store
    #             errorMat[i, k] = self.rssError(yEst.T.A, np.array(testY))
    #             # print errorMat[i,k]
    #     errorMat=pd.DataFrame(errorMat)
    #     errorMat.fillna(0)
    #     errorMat = errorMat.values
    #     meanErrors = np.mean(errorMat, 0)  # calc avg performance of the different ridge weight vectors
    #     minMean = float(min(meanErrors))
    #     bestWeights = wMat[np.nonzero(meanErrors == minMean)]
    #     # can unregularize to get model
    #     # when we regularized we wrote Xreg = (x-meanX)/var(x)
    #     # we can now write in terms of x not Xreg:  x*w/var(x) - meanX/var(x) +meanY
    #     xMat = np.mat(xArr)
    #     yMat = np.mat(yArr).T
    #     meanX = np.mean(xMat, 0)
    #     varX = np.var(xMat, 0)
    #     unReg = bestWeights / varX
    #     #print("the best model from Ridge Regression is:\n", unReg, bestWeights)
    #     #print("with constant term: ", -1 * sum(np.multiply(meanX, unReg)) + np.mean(yMat))
    #     #print(bestWeights)
    #     return bestWeights

    # 交叉验证测试
    def cross_validation_test(self):
        abX, abY = self.load_model_data()
        #abX = np.mat(abX)
        #abY = np.mat(abY)
        bestWeights = self.cross_validation(abX, abY)
        return bestWeights




if __name__=="__main__":
    obj = Model()
    #obj.loadDataSet_test()
    xMat = obj.load_test_final_data()
    ws = obj.cross_validation_test()
    #print(ws)
    xMat = np.mat(xMat)
    ws = np.mat(ws).T
    yHat = xMat * ws
    result_df = pd.DataFrame(columns=["Id", "SalePrice"])
    print("--------------------")
    result_df["Id"] = obj.ids
    result_df["SalePrice"] = np.array(yHat.T)
    result_df.to_csv("test_reset.csv")