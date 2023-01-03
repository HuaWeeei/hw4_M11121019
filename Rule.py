# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 00:24:23 2022

@author: bbill
"""

from mlxtend.frequent_patterns import  apriori,association_rules
import numpy as np
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
import time
# with tf.device('/gpu:0'):
df=pd.read_csv("Store.csv")
X = df.iloc[:,0:7]
# Gather All Items of Each Transactions into Numpy Array
transaction = []
for i in range(0, X.shape[0]):
    for j in range(0, X.shape[1]):
        transaction.append(X.values[i,j])
# converting to numpy array
transaction = np.array(transaction)
# print(transaction)
#  Transform Them a Pandas DataFrame
df = pd.DataFrame(transaction, columns=["items"]) 
# Put 1 to Each Item For Making Countable Table, to be able to perform Group By
df["incident_count"] = 1 
#  Delete NaN Items from Dataset
indexNames = df[df['items'] == "nan" ].index
df.drop(indexNames , inplace=True)
# Making a New Appropriate Pandas DataFrame for Visualizations  
df_table = df.groupby("items").sum().sort_values("incident_count"
                                                 , ascending=False).reset_index()
#  Initial Visualizations
df_table.head(5).style.background_gradient(cmap='Blues')
print('購買次數前10：',df_table)
transaction = []
for i in range(X.shape[0]):
    transaction.append([str(X.values[i,j]) for j in range(X.shape[1])])
# creating the numpy array of the transactions
transaction = np.array(transaction)
# initializing the transactionEncoder 資料轉換one-hot
te = TransactionEncoder()
te_ary = te.fit(transaction).transform(transaction)
dataset = pd.DataFrame(te_ary, columns=te.columns_)
dataset = dataset.drop("nan",axis=1)
# select top 30 items 選擇前30的值
# first30 = df_table["items"].head(30).values 
# dataset = dataset.loc[:,first30] 
from mlxtend.frequent_patterns import fpgrowth
tStart_fpgrowth = time.time()
#running the fpgrowth algorithm
res=fpgrowth(dataset,min_support=0.01, use_colnames=True
             )
ftp = association_rules(res,metric='lift',min_threshold=1)
tEnd_fpgrowth = time.time()
ftp = ftp.sort_values("confidence",ascending=False)
# ftp.to_csv('fp-growth001.csv',index = False)
print("fpgrowth",ftp)
#running the apriori algorithm
tStart_apr = time.time()
ap_items = apriori(dataset,min_support=0.01 ,use_colnames=True
                   )
rules = association_rules(ap_items,metric='lift',min_threshold=1)
rules = rules.sort_values("confidence",ascending=False)
# rules.to_csv('apriori001.csv',index = False)
print(rules)
# association_results = pd.DataFrame(association_rules)  
tEnd_apr = time.time()               
df_c = pd.DataFrame({'fpg': [tEnd_fpgrowth - tStart_fpgrowth ],
                      'aprioi': [tEnd_apr - tStart_apr ]
                          }) 
print("Time：",df_c)
#-----------------------輸入rules該有的商品則能找出對應商品
count = len(rules['lift'])
itemse= rules['antecedents']
con = rules['consequents']
pro=[]
print("商品清單：")
for m in range(count):
    print(rules['antecedents'][m])
p=str(input("請輸入上表商品組合：")) 
#op = str("frozenset({'") + p +str("})")
for i in range(count):
    if  p == str(itemse[i]) :
        recon = str(con[i])
        recon = recon.replace("frozenset({","")
        recon = recon.replace("})","")
        recon = recon.replace("'","")
        recon = recon.replace("'","")
        recon = recon.replace(" ","")
        pro.append(recon)
pro=','.join(pro) #list以逗號為分界轉字串_格式轉
pro_list = pro.split(',')  #以逗號為分界轉list
if pro_list==['']:
    print("尚無推薦商品類型")
else:  
    pro_ar = np.array(pro_list)
    pro_ar = np.unique(pro_ar)
    print("推薦商品類型為：",pro_ar)




