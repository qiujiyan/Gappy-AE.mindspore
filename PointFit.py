from ctypes import sizeof
import numpy as np 
import sys


from util import *
from sklearn import linear_model

class PointFit():
    def __init__(self,NorFiled=None,Filed=None,deg=3,Chebyshev=False):
        if NorFiled is not None :
            self.n=NorFiled
        else:
            self.n = getNorFiled()

        if Filed is not None :
            self.f=Filed
        else:
            self.f = getFiled()
        
        self.x,self.y = getXY()

        self.dataSize = len(self.n)
        self.meshSize = len(self.x)

        p=[i[2][0] for i in self.n]
        self.p = np.array(p).T

        u=[i[2][1] for i in self.n]
        self.u = np.array(u).T

        v=[i[2][2] for i in self.n]
        self.v = np.array(v).T
        
        aoamach = [[i,j] for i,j,k in self.n]
        self.aoamach = np.array(aoamach)
        self.aoa,self.mach = self.aoamach.T
        
        self.deg=deg
        self.degs=0
        self.Chebyshev =Chebyshev
        self.X = self.getX(self.aoa,self.mach,self.degs,self.deg,Chebyshev =  self.Chebyshev)
        
    def getX(self,a,b,dege=0,deg=3,Chebyshev=False):
        if Chebyshev:
            Xa = [np.cos(n*np.arccos(a/180*np.pi)) for n in range(dege,deg)]
            Xb = [np.cos(n*np.arccos(b)) for n in range(dege,deg)]
        else :
            Xa = [a**n for n in range(dege,deg)]
            Xb = [b**n for n in range(dege,deg)]
        X=[]
        self.Xa = Xa
        self.Xb = Xb
        
        for i in Xa:
            for j in Xb:
                X.append(i*j)
#         X = X[1:]
        return  np.array(X).T




    

    def getBestPoint(self,number,path="puvLinearRegression.npz"):
        try:
            return self.pickPUV(number)
        except:
            pass

        try:
            self.laod(path)
            return self.pickPUV(number)
        except:
            pass

        try:
            self.fitPUV()
            return self.pickPUV(number)
        except:
            pass 


    def getBestPointMap(self,number,path="puvLinearRegression.npz"):
        try:
            bestPoint =  self.getBestPoint(number,path)
            maskMap = np.zeros(self.meshSize)
            maskMap[bestPoint] = 1
            return maskMap.copy()
        except:
            pass

        
        
    def getBestPointPUVNumber(self,n1,n2,n3,path="puvLinearRegression.npz"):
        try:
            return self.pickP(n1),self.pickU(n2),self.pickV(n3)
        except:
            pass

        try:
            self.laod(path)
            return self.pickP(n1),self.pickU(n2),self.pickV(n3)
        except:
            pass

        try:
            self.fitPUV()
            return self.pickP(n1),self.pickU(n2),self.pickV(n3)
        except:
            pass 

        
    def fit_and_save(self):
        self.fitPUV()
        self.saveLinearRegression()

    def laod(self,path = "puvLinearRegression.npz"):
        dataall = np.load(path)
        index=0
        self.pScore = dataall['arr_%d'%index];index = index+1
        self.pCoef = dataall['arr_%d'%index];index = index+1
        self.pIntercept = dataall['arr_%d'%index];index = index+1
        self.uScore = dataall['arr_%d'%index];index = index+1
        self.uCoef = dataall['arr_%d'%index];index = index+1
        self.uIntercept = dataall['arr_%d'%index];index = index+1
        self.uvScore = dataall['arr_%d'%index];index = index+1
        self.uvCoef = dataall['arr_%d'%index];index = index+1
        self.uvIntercept = dataall['arr_%d'%index];index = index+1
   
        

    def saveLinearRegression(self,path = "puvLinearRegression.npz"):
        savelist = [self.pScore,self.pCoef,self.pIntercept,self.uScore,self.uCoef,self.uIntercept,self.uvScore,self.uvCoef,self.uvIntercept]
        np.savez(path,*savelist)

    def fitPUV(self):
        self.pScore,self.pCoef,self.pIntercept =  self.fit(self.p)
        self.uScore,self.uCoef,self.uIntercept =  self.fit(self.u)
        self.uvScore,self.uvCoef,self.uvIntercept =  self.fit(self.v)


    def pickPUV(self,number):
        self.rank = np.argsort(-self.pScore-self.uScore-self.uvScore)
        self.rankIndex=  np.argwhere(self.rank<number)
        self.rankBest =  self.rank[self.rankIndex]
        return self.rankBest.flatten().copy()

    def pickP(self,number):
        self.rankP = np.argsort(-self.pScore)
        self.rankIndexP =  np.argwhere(self.rankP<number)
        self.rankBestP =  self.rankP[self.rankIndexP]
        return self.rankBestP.copy().flatten()




    def pickU(self,number):
        self.rankU = np.argsort(-self.uScore)
        self.rankIndexU=  np.argwhere(self.rankU<number)
        self.rankBestU =  self.rankU[self.rankIndexU]
        return self.rankBestU.copy().flatten()

    def pickV(self,number):
        self.rankV = np.argsort(-self.uvScore)
        self.rankIndexV=  np.argwhere(self.rankV<number)
        self.rankBestV =  self.rankV[self.rankIndexV]
        return self.rankBestV.copy().flatten()

    def maskP(self,number):
        pick = self.pickP(number)
        maskMap = np.zeros(self.meshSize)
        maskMap[pick] = 1
        return maskMap.copy()

    def maskU(self,number):
        pick = self.pickU(number)
        maskMap = np.zeros(self.meshSize)
        maskMap[pick] = 1
        return maskMap.copy()

    def maskV(self,number):
        pick = self.pickV(number)
        maskMap = np.zeros(self.meshSize)
        maskMap[pick] = 1
        return maskMap.copy()


    def reversPUV(self,aoaRev,machRev):
#         XRev=[aoaRev,machRev,aoaRev*machRev,aoaRev*aoaRev,machRev*machRev]
#         XRev = np.array(XRev).T

        
#         Xa = np.array([1,aoaRev,aoaRev**2,aoaRev**3])
#         Xm = np.array([1,machRev,machRev**2,machRev**3])
#         X=[]
#         for i in Xa:
#             for j in Xm:
#                 X.append(i*j)
#         XRev = np.array(X[1:])
#         XRev = XRev.T
        XRev= self.getX(aoaRev,machRev,self.degs,self.deg,Chebyshev =  self.Chebyshev)
        if (len(XRev.shape)<2):
            XRev = np.array([XRev])
        self.pRevers = self.revers(self.pCoef,self.pIntercept,XRev)
        self.uRevers = self.revers(self.uCoef,self.uIntercept,XRev)
        # self.uvRevers = self.revers(self.uvCoef,self.uvIntercept,XRev)
        # self.vRevers  = self.uRevers/self.uvRevers
        self.vRevers = self.revers(self.uvCoef,self.uvIntercept,XRev)
        # v必须先拟合再归一化
        
        # nm,ni = getMaxMin()
        # _ ,_ ,self.vRevers = Normalization((self.pRevers ,self.uRevers ,self.vRevers),nm,ni)
        
        
        return self.pRevers.copy(),self.uRevers.copy(),self.vRevers.copy()

    def fit(self,target):
        scoreList=np.zeros(self.meshSize)
        coefList=np.zeros((self.meshSize,len(self.X[0])))
        interceptList=np.zeros(self.meshSize)
        for position in range( self.meshSize):
            Y=target[position]
            Y=np.array(Y)
            regr = linear_model.LinearRegression()
            res = regr.fit(self.X, Y)
            coefList[position] = res.coef_
            interceptList[position] = res.intercept_
            scoreList[position] = res.score(self.X,Y)
        return scoreList.copy(),coefList.copy(),interceptList.copy()

    def revers(self,coef,intercept,XRev):
        d=  XRev@coef.T
        d=  d+ np.tile(intercept, (d.shape[0],1))
        return  d.copy()


if __name__ == '__main__':
    pf =  PointFit()
    pf.getBestPoint(100)