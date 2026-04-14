#ANÁLISE EXPLORATÓRIA DE DADOS - EDA

import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt

df = pd.read_csv("C:\\Users\\Rumanns\\Desktop\\cell phone project analyse\\data\\Amazon_Cell_Phones.csv")


print(df.columns)

#print(df['cpu_model'].unique())

q90_vendas = df['ID'].quantile(0.90)

print(q90_vendas)

print('Perfequito!')

#for col in df.columns:
#	print(df[col].unique())


