#-*- coding: utf-8 -*-

import pandas as pd

def result():
    df = pd.read_csv("result.csv")
    df = df.drop(columns=["Id"])
    df["Id"] = df["id"].copy()
    df = df[["Id", "SalePrice"]]
    print(df)

if __name__=="__main__":
    result()