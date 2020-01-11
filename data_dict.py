#-*- coding: utf-8 -*-

MSZoning_dict = {
            "A": 1,
            "C": 2,
            "FV": 3,
            "I": 4,
            "RH": 5,
            "RL": 6,
            "RP": 7,
            "RM": 8,
            "C (all)":2,
        }
Street_dict = {
            "Grvl": 1,
            "Pave": 2
        }

Alley_dict = {
            "Grvl": 1,
            "Pave": 2,
            "NA": 3
        }

LotShape_dict = {
            "Reg": 1,
            "IR1": 2,
            "IR2": 3,
            "IR3": 4
        }

LandContour_dict = {
            "Lvl": 1,
            "Bnk": 2,
            "HLS": 3,
            "Low": 4
        }

Utilities_dict = {
            "AllPub": 1,
            "NoSewr": 2,
            "NoSeWa": 3,
            "ELO": 4
        }

LotConfig_dict = {
            "Inside": 1,
            "Corner": 2,
            "CulDSac": 3,
            "FR2": 4,
            "FR3": 4
        }

LandSlope_dict = {
            "Gtl": 1,
            "Mod": 2,
            "Sev": 3
        }

Neighborhood_dict = {
            "Blmngtn": 1,
            "Blueste": 2,
            "BrDale": 3,
            "BrkSide": 4,
            "ClearCr": 5,
            "CollgCr": 6,
            "Crawfor": 7,
            "Edwards": 8,
            "Gilbert": 9,
            "IDOTRR": 10,
            "MeadowV": 11,
            "Mitchel": 12,
            "Names": 13,
            "NAmes":13,
            "NoRidge": 14,
            "NPkVill": 15,
            "NridgHt": 16,
            "NWAmes": 17,
            "OldTown": 18,
            "SWISU": 19,
            "Sawyer": 20,
            "SawyerW": 21,
            "Somerst": 22,
            "StoneBr": 23,
            "Timber": 24,
            "Veenker": 25
        }

Condition1_dict = {
    "Artery": 1,
    "Feedr": 2,
    "Norm": 3,
    "RRNn": 4,
    "RRAn": 5,
    "PosN": 6,
    "PosA": 7,
    "RRNe": 8,
    "RRAe": 9
}

Condition2_dict = {
    "Artery": 1,
    "Feedr": 2,
    "Norm": 3,
    "RRNn": 4,
    "RRAn": 5,
    "PosN": 6,
    "PosA": 7,
    "RRNe": 8,
    "RRAe": 9
}

BldgType_dict = {
    "1Fam": 1,
    "2FmCon": 2,
    "2fmCon":2,
    "Duplx": 3,
    "Duplex":3,
    "TwnhsE": 4,
    "TwnhsI": 5,
    "Twnhs":4.5,
}

HouseStyle_dict = {
    "1Story": 1,
    "1.5Fin": 2,
    "1.5Unf": 3,
    "2Story": 4,
    "2.5Fin": 5,
    "2.5Unf": 6,
    "SFoyer": 7,
    "SLvl": 8
}

RoofStyle_dict = {
    "Flat": 1,
    "Gable": 2,
    "Gambrel": 3,
    "Hip": 4,
    "Mansard": 5,
    "Shed": 6
}

RoofMatl_dict = {
    "ClyTile": 1,
    "CompShg": 2,
    "Membran": 3,
    "Metal": 4,
    "Roll": 5,
    "Tar&Grv": 6,
    "WdShake": 7,
    "WdShngl": 8
}

Exterior1st_dict = {
        "AsbShng":1,
        "AsphShn":2,
        "BrkComm":3,
        "BrkFace":4,
        "CBlock":5,
        "CemntBd":6,
        "HdBoard":7,
        "ImStucc":8,
        "MetalSd":9,
        "Other":10,
        "Plywood":11,
        "PreCast":12,
        "Stone":13,
        "Stucco":14,
        "VinylSd":15,
        "Wd Sdng":16,
        "WdShing":17
}

Exterior2nd_dict = {
        "AsbShng":1,
        "AsphShn":2,
        "BrkComm":3,
        "BrkFace":4,
        "CBlock":5,
        "CemntBd":6,
        "HdBoard":7,
        "ImStucc":8,
        "MetalSd":9,
        "Other":10,
        "Plywood":11,
        "PreCast":12,
        "Stone":13,
        "Stucco":14,
        "VinylSd":15,
        "Wd Sdng":16,
        "WdShing":17,
        "Wd Shng":17,
        "Brk Cmn":3,
        "CmentBd":6,
}

MasVnrType_dict = {
    "BrkCmn":1,
    "BrkFace": 2,
    "CBlock": 3,
    "None": 4,
    "Stone": 5
}

ExterQual_dict = {
    "Ex":1,
    "Gd":2,
    "TA":3,
    "Fa":4,
    "Po":5,
    "NA":6
}

ExterCond_dict = {
    "Ex":1,
    "Gd":2,
    "TA":3,
    "Fa":4,
    "Po":5,
    "NA": 6
}

Foundation_dict = {
    "BrkTil": 1,
    "CBlock": 2,
    "PConc": 3,
    "Slab": 4,
    "Stone": 5,
    "Wood": 6
}

BsmtQual_dict = {
    "Ex":1,
    "Gd":2,
    "TA":3,
    "Fa":4,
    "Po":5,
    "NA": 6
}

BsmtCond_dict = {
    "Ex":1,
    "Gd":2,
    "TA":3,
    "Fa":4,
    "Po":5,
    "NA": 6
}

BsmtExposure_dict = {
    "Gd":1,
    "Av":2,
    "Mn":3,
    "No":4,
    "NA":5
}

BsmtFinType1_dict = {
    "GLQ":1,
    "ALQ":2,
    "BLQ":3,
    "Rec":4,
    "LwQ":5,
    "Unf":6,
    "NA": 7
}

BsmtFinType2_dict = {
    "GLQ":1,
    "ALQ":2,
    "BLQ":3,
    "Rec":4,
    "LwQ":5,
    "Unf":6,
    "NA": 7
}

Heating_dict = {
    "Floor": 1,
    "GasA": 2,
    "GasW": 3,
    "Grav": 4,
    "OthW": 5,
    "Wall": 6
}

HeatingQC_dict = {
    "Ex":1,
    "Gd":2,
    "TA":3,
    "Fa":4,
    "Po":5,
    "NA": 6
}

CentralAir_dict = {
    "N":0,
    "Y":1,
}

Electrical_dict = {
    "SBrkr": 1,
    "FuseA": 2,
    "FuseF": 3,
    "FuseP": 4,
    "Mix": 5
}

KitchenQual_dict = {
    "Ex":1,
    "Gd":2,
    "TA":3,
    "Fa":4,
    "Po":5,
    "NA": 6
}

Functional_dict = {
    "Typ": 1,
    "Min1": 2,
    "Min2": 3,
    "Mod": 4,
    "Maj1": 5,
    "Maj2": 6,
    "Sev": 7,
    "Sal": 8
}

FireplaceQu_dict = {
    "Ex": 1,
    "Gd": 2,
    "TA": 3,
    "Fa": 4,
    "Po": 5,
    "NA": 6
}

GarageType_dict = {
    "2Types": 1,
    "Attchd": 2,
    "Basment": 3,
    "BuiltIn": 4,
    "CarPort": 5,
    "Detchd": 6,
    "NA": 7
}

GarageFinish_dict = {
    "Fin": 1,
    "RFn": 2,
    "Unf": 3,
    "NA": 4
}

GarageQual_dict = {
    "Ex":1,
    "Gd":2,
    "TA":3,
    "Fa":4,
    "Po":5,
    "NA": 6
}

GarageCond_dict = {
    "Ex": 1,
    "Gd": 2,
    "TA": 3,
    "Fa": 4,
    "Po": 5,
    "NA": 6
}

PavedDrive_dict = {
    "Y": 1,
    "P": 2,
    "N": 3
}

PoolQC_dict = {
    "Ex": 1,
    "Gd": 2,
    "TA": 3,
    "Fa": 4,
    "Po": 5,
    "NA": 6
}

Fence_dict = {
    "GdPrv": 1,
    "MnPrv": 2,
    "GdWo": 3,
    "MnWw": 4,
    "NA": 5
}

MiscFeature_dict = {
    "Elev": 1,
    "Gar2": 2,
    "Othr": 3,
    "Shed": 4,
    "TenC": 5,
    "NA": 6
}

SaleType_dict = {
    "WD": 1,
    "CWD": 2,
    "VWD": 3,
    "New": 4,
    "COD": 5,
    "Con": 6,
    "ConLw": 7,
    "ConLI": 8,
    "ConLD": 9,
    "Oth": 10
}

SaleCondition_dict = {
    "Normal": 1,
    "Abnorml": 2,
    "AdjLand": 3,
    "Alloca": 4,
    "Family": 5,
    "Partial": 6
}