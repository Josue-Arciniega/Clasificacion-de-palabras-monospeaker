import numpy as np
import matplotlib.pyplot as plt 
class KM():
    def kmeans2(self,X,Y,clusters,th):
        datos=np.matrix(X).T
        #U=np.matrix('0 0 1 1 ; 1 1 0 0')
        U=np.matrix(np.zeros((clusters,datos.shape[1])))
        for i in range(datos.shape[1]):
            U[:,i][Y[i]]=1
        V=np.matrix(np.zeros((U.shape[0],datos.shape[0])))
        D=np.copy(U).astype(np.float32)
        for k in range(40):
            Ua=np.matrix(np.copy(U))
            for i in range(V.shape[0]):
                for j in range (V.shape[1]):
                    V[i,j]=float(U[i]*datos[j].T)/np.sum(U[i])
            for i in range(D.shape[0]):
                for j in range (D.shape[1]):
                    #print(U[i],datos[i])
                    D[i,j]=np.sqrt(sum(np.array((datos[:,j]-V[i].T))**2))
            for i in range(D.shape[1]):
                Ua[:,i]=np.zeros((U.shape[0],1))
                Ua[:,i][np.argmin(D[:,i])]=1
            if(np.sum(np.abs(Ua-U))<th):
                print("Kmeans iteraciones: "+str(k))
                break
            U=Ua
        return V

    def predict(self,X,V):
        clusters=V.shape[0]
        datos=np.matrix(X).T
        pred=[]
        U=np.matrix(np.zeros((clusters,datos.shape[1])))
        D=np.copy(U).astype(np.float32)
        for i in range(D.shape[0]):
            for j in range (D.shape[1]):
                D[i,j]=np.sqrt(sum(np.array((datos[:,j]-V[i].T))**2))                  
        for i in range(D.shape[1]):
            pred.append(np.argmin(D[:,i]))
        return np.array(pred)
              

