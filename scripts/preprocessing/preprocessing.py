from scripts.preprocessing.preprocessing_list import PREPROCESSING_LIST

def Preprocessing(name):
    preprocessing = PREPROCESSING_LIST[name]
    return preprocessing

