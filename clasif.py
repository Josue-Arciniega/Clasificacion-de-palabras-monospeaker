import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame
data=pd.read_csv("data.csv")

def mapeaClases(y):
    a=[]
    for clase in y:
        if clase=='oaxaca':
            a.append(0)
        elif clase=='michi':
            a.append(1)
        elif clase=='circunscrito':
            a.append(2)
    return np.array(a)
mc=lambda s:0 if s=='oaxaca' else (1 if s=='michi'  else 2)
X=data.iloc[:,0:-1].values
X[:,0]=X[:,0]/10000
y=data.iloc[:,-1].values
y=mapeaClases(y)
data['c1']=X[:,0]
kmtra=[]
kmtes=[]
cmtra=[]
cmtes=[]
for p in range(20):
    print("Particionado " +str(p))
    if p not in [4,9,11,13,14,15,17,18,19]:
        from sklearn.model_selection import train_test_split
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.2,random_state=p,stratify=y)

        datosrend={}
        from  kmeans import KM
        km=KM()
        V=km.kmeans2(X_train,y_train,3,.001)
        pred_test=km.predict(X_test,V)
        pred_train=(km.predict(X_train,V))

        aciertorate_train=np.sum(pred_train==y_train)/y_train.shape[0]
        aciertorate_test=np.sum(pred_test==y_test)/y_test.shape[0]

        from cmeans import CM
        cm=CM(6)
        Vc=cm.cmeans(X_train,y_train,3,.001)

        pred_test_c=cm.predict(X_test,Vc)
        pred_train_c=(cm.predict(X_train,Vc))

        aciertorate_train_c=np.sum(pred_train_c==y_train)/y_train.shape[0]
        aciertorate_test_c=np.sum(pred_test_c==y_test)/y_test.shape[0]


        print("Kmeans_train "+str(aciertorate_train)+"%")
        print("Kmeans_test "+str(aciertorate_test)+"%")
        print("Cmeans_train "+str(aciertorate_train_c)+"%")
        print("Cmeans_test "+str(aciertorate_test_c)+"%")
        
        kmtra.append(aciertorate_train)
        kmtes.append(aciertorate_test)
        cmtra.append(aciertorate_train_c)
        cmtes.append(aciertorate_test_c)




fig, ax = plt.subplots()
colors={0:'blue',1:'red',2:'green'}
df=data
agrupado=df.groupby('clase')
for key, grupo in agrupado:
    grupo.plot(ax=ax,x='c1',y='c2',kind='scatter',color=colors[mc(key)])
       
plt.scatter(Vc[:,0].tolist(),Vc[:,1].tolist(),color='purple',marker="*",linewidths=7)
plt.scatter(V[:,0].tolist(),V[:,1].tolist(),color='orange',marker="*",linewidths=7)

plt.legend(["circunscrito","michi","oaxaca","C-means Centroids","K-means Centroidsi"])

plt.show()  

print("Eficiencia promedio Kmeans_train "+str(np.mean(kmtra))+"%")
print("Eficiencia pŕomedio Kmeans_test "+str(np.mean(kmtes))+"%")
print("Eficiencia promedio Cmeans_train "+str(np.mean(cmtra))+"%")
print("Eficiencia promedio Cmeans_test "+str(np.mean(cmtes))+"%")
        

