import numpy as np
import matplotlib.pyplot as plt 
class CM():
    def __init__(self,ordm):
            # body of the constructor
        self.m=ordm
    def cmeans(self,X,Y,clusters,th):
        m=self.m
        itera=0
        datos=np.matrix(X).T
        #U=np.matrix('0 0 1 1 ; 1 1 0 0')
        U=np.matrix(np.ones((clusters,datos.shape[1])))
        for i in range(datos.shape[1]):
            U[:,i][Y[i]]=3000        
        U=U/3002
        V=np.matrix(np.zeros((U.shape[0],datos.shape[0])))
        V=np.matrix(np.random.random((U.shape[0],datos.shape[0])))*10
        D=np.copy(U).astype(np.float32)
        
        for k in range(300):
            Ua=np.matrix(np.copy(U))
            for i in range(V.shape[0]):
                    for j in range (V.shape[1]):
                        V[i,j]=float((np.array(U[i])**2)*datos[j].T)/np.sum(U[i])
            
            for i in range(D.shape[0]):
                    for j in range (D.shape[1]):
                        #print(U[i],datos[i])
                        D[i,j]=np.sqrt(sum(np.array((datos[:,j]-V[i].T))**2))
            Ua=np.zeros((D.shape[0],D.shape[1]))
            for i in range(U.shape[0]):
                for j in range(U.shape[1]):
                    Ua[i,j]=np.sum((D[i,j]**m)*(np.array(D[:,j])**-m))                      
            Ua=Ua**-1
            itera+=1
            #print(np.sum(np.abs(Ua-U)))
            if(np.sum(np.abs(Ua-U))<th):
                    break
            U=Ua
        print("Cmeans iteaciones:" +str(itera))
        return V

    def predict(self,X,V):
        datos=np.matrix(X).T
        Ua=np.matrix(np.ones((V.shape[0],datos.shape[1])))
        D=np.copy(Ua).astype(np.float32)
        m=self.m
        pred=[]
        for i in range(D.shape[0]):
            for j in range (D.shape[1]):
             #print(U[i],datos[i])
                 D[i,j]=np.sqrt(sum(np.array((datos[:,j]-V[i].T))**2))
            Ua=np.zeros((D.shape[0],D.shape[1]))
            for i in range(Ua.shape[0]):
                for j in range(Ua.shape[1]):
                    Ua[i,j]=np.sum((D[i,j]**m)*(np.array(D[:,j])**-m))                      
            Ua=Ua**-1
        for i in range(Ua.shape[1]): 
            pred.append(np.argmax(Ua[:,i])) 
        return np.array(pred)

