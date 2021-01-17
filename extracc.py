import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as waves
import os
from pandas import DataFrame
from scipy.fft import fft
Hh=lambda F,A:-1*np.sum((F+A)*np.log2(F+A))
#Hv=lambda F,A:-1*np.sum((F*A)*np.log2(A*F))
Hv=lambda F,A:np.sum(np.corrcoef(A,F))
th=30
dg='./palabras'
direc=dg
c1=[]
c2=[]
cla=[]
color=['r','g','b']
contenido = os.listdir(direc)
for p in contenido:
    plt.figure(p)
    for f in os.listdir(direc+'/'+p):
        plt.subplot(211)
        print('{}/{}/{}'.format(direc,p,f))
        #importando wav file
        fm,audio = waves.read('{}/{}/{}'.format(direc,p,f))
        if len(audio.shape)==2:
            audio=audio[:,0]
        audio=(audio[1::6]/np.max(audio))
        plt.plot(audio)
        plt.subplot(212)
        fr=np.linspace(0,4000,int(len(audio)/2))
        espectro=np.abs(fft(audio)[0:int(len(audio)/2)])
        espectro/=np.max(espectro)
        fr[0]=.001
        c1.append(Hh(fr,espectro))
        c2.append(Hv(fr,espectro))
        cla.append(p)
        plt.plot(fr,espectro)
        plt.xlabel('Frec(Hz)')

plt.show()
df=DataFrame(dict(c1=c1,c2=c2,clase=cla))
colors={0:'blue',1:'red',2:'green'}
fig, ax = plt.subplots()
agrupado=df.groupby('clase')
for key, grupo in agrupado:
    grupo.plot(ax=ax,x='c1',y='c2',kind='scatter',label=key,color=colors[contenido.index(key)])
plt.show()      
df.to_csv("data.csv",index=False)









