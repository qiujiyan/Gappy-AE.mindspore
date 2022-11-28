#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
ZetCode PyQt5 tutorial 

This example shows a tooltip on 
a window and a button.

Author: Jan Bodnar
Website: zetcode.com 
Last edited: August 2017
"""

import sys
from PyQt5.QtWidgets import (QWidget, QToolTip, 
    QPushButton, QApplication,QTextEdit,QLineEdit,QLabel,QDoubleSpinBox )
from PyQt5.QtGui import QFont    
import argparse
import os
from util import  *
from PointFit import  *
import _thread

def plotData(aoa,m,path1,path2):
    import mindspore as ms
    from mindspore import dataset as ds
    from mindspore import Tensor
    import mindspore.nn as nn
    import numpy as np
    from scipy.optimize import fmin_cg
    from model.mlp import MLP
    ms.set_context(mode=ms.PYNATIVE_MODE)

    class Namespace:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    args = Namespace()
    args = Namespace(batch_size=10, data_size=20052, epochs=100, fit_deg=5, learning_rate=0.001, loss_train='mse', mask_num=3000, mod_number=32, model='mlp', output_dir='./log/mlp_modNumber32', repeat=1, save_freq=50, task='train and test', test_size=0.33)
    args.fit_deg = 5
    # args.mod_number = 120
    args.mask_num = 6000
    args.model_path = path2 #'log/net.ckpt'
    args.input_size = 6684*3

    point_fiter  = PointFit(deg=args.fit_deg)
    point_fiter.laod(path1 ) #'log/PointFit_5deg.npz'

    net = MLP(args)
    ms.load_checkpoint(args.model_path,net)


    pr,ur,vr = point_fiter.reversPUV(aoa,m)
    maskP = point_fiter.maskP(args.mask_num)
    maskU = point_fiter.maskU(args.mask_num)
    maskV = point_fiter.maskV(args.mask_num)

    puvr = np.concatenate([pr,ur,vr]).astype(np.float32).flatten()
    puvr = puvr.reshape((1,puvr.shape[0]))
    puvr = Tensor(puvr)
    
    mask_map = np.concatenate([maskP,maskU,maskV]).astype(np.float32).flatten()
    mask_map = mask_map.reshape((1,mask_map.shape[0]))
    mask_map = Tensor(mask_map)
    
    f = net.loss_decoder_helper(puvr,mask_map)
    fp= net.grad_loss_decoder_helper(puvr,mask_map)
    start_code =np.zeros(args.mod_number).astype(np.float32)
    fmin_code =fmin_cg(f,start_code,fprime=fp,disp=False)
    y_per = net.decoder( Tensor(fmin_code.astype(np.float32)).reshape((1,fmin_code.shape[0])))
    
    x,y = getXY()
    import matplotlib.pyplot as plt
    plt.figure()
    plt.scatter(x,y ,c=y_per[0,0:6684],s=0.1)
    plt.xlim(-0.5,1.5)
    plt.ylim(-1,1)
    plt.show()
    # print('y_per',y_per)
    return y_per

class Example(QWidget):

    def __init__(self):
        super().__init__()
        self.initUI()


    def initUI(self):
        QToolTip.setFont(QFont('SansSerif', 10))

        labelt1 = QLabel("PointFit.npz",self)
        labelt1 .move(50,35)

        self.mod1 = QLineEdit('log/PF.npz',self)
        self.mod1.move(170,30)



        labelt1 = QLabel("net.ckpt",self)
        labelt1.move(50,85)

        self.mod2 = QLineEdit("log/net.ckpt",self)
        self.mod2.move(170,80)

        label1 = QLabel("AOA",self)
        label1.move(50,155)

        self.number1 = QDoubleSpinBox (self)
        self.number1.move(100,150)
        self.number1.setRange(-10,10)
        self.number1.setSingleStep(0.5)

        label2 = QLabel("mach",self)
        label2.move(200,155)

        self.number2 = QDoubleSpinBox (self)
        self.number2.move(250,150)
        self.number2.setRange(0,1)
        self.number2.setSingleStep(0.05)
        
        btn = QPushButton('Run', self)
        btn.move(350, 150)
        btn.clicked.connect(self.run)

        self.setGeometry(300, 300, 500, 200)
        self.setWindowTitle('NACA0012')    
        self.show()

    def run(self):

        aoa = float(self.number1.value ())
        m = float(self.number2.value ())
        p1 = self.mod1.text () 
        p2 = self.mod2.text () 
        plotData(aoa, m,p1,p2 )

        

if __name__ == '__main__':

    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())