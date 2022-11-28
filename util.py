import pickle
import os
import numpy as np

abspath = os.path.dirname(__file__)
datapath =  os.path.join(abspath ,'NACA0012_DATA')


# 读取path对应的pkl文件
def readPkl(path):
    return pickle.load(open(path,'rb'))

# 读取 train_max_min.pkl文件，获取最大最小值，以便归一化。
def getMaxMin():
    return readPkl(os.path.join(datapath,'train_max_min.pkl'))


# 读取 读取全部流场pkl文件，返回列表[ re,mach,[]]
def getFiled():
    dataList = os.listdir(os.path.join(datapath,'dataset'))
    def getInfo(i):
        allPath = os.path.join(datapath,'dataset',i)
        aoa = float(i.split('_')[4])
        mach = float(i.split('_')[6].split('.pkl')[0])
#         print('aoa',aoa,'mach',mach)

        return aoa,mach,readPkl(allPath)

    return [getInfo(i) for i in dataList]



"""
将data中的u,v转化成极坐标 xita , ro ，p不变。
"""
def polar(data):
    v= data[1]
    u= data[2]
    Theta = np.arctan(v/u)
    ro = np.sqrt(v**2+u**2)
    return [data[0],Theta,ro]


def Normalization(data,nm,ni):
    return [(f-mi)*2/(ma-mi)-1 for ma,mi,f in zip(nm,ni,data)]
    
    
"""
getFiled之后，对 p,v,u 进行归一化
"""
def getNorFiled():

    nm,ni = getMaxMin()
    return [[i,j,Normalization(data,nm,ni)] for i,j,data in getFiled()]

"""
getFiled之后，将速度u,v转化成极坐标 xita , ro 
"""
def getFiledPolar():
    return [[i,j,polar(data)] for i,j,data in getFiled()]


"""
从XY.csv读取网格点的x,y坐标
Typical usage example:
x,y = getXY()
"""
def getXY():
    import pandas as pd
    f = pd.read_csv(os.path.join(datapath,'XY.csv'))
    return f[['x','y']].to_numpy().T



if __name__ == '__main__':
    print(getMaxMin())
    print(getFiled())