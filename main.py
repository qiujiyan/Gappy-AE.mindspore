import argparse
import os
import mindspore as ms
from mindspore import dataset as ds
from mindspore import Tensor
import mindspore.nn as nn
import numpy as np
from scipy.optimize import fmin_cg
from util import  *
from PointFit import  *


def log(*args,**kw):
    print(*args,**kw,flush = True)

def get_args():
    parser = argparse.ArgumentParser('ArgumentParser', add_help=False)
    #task
    parser.add_argument('--task',  type=str, default='train and test')

    # model parameters
    parser.add_argument('--model',  type=str, default='mlp')
    parser.add_argument('--mod_number',  type=int, default=120)
    parser.add_argument('--data_size',  type=int, default=6684*3)
    
    #test parameters
    parser.add_argument('--model_path',  type=str, default=None)
    parser.add_argument('--point_fit_path',  type=str, default=None)


    # training parameters
    parser.add_argument('--test_size',  type=float, default=0.33)
    parser.add_argument('--learning_rate',  type=float, default=1e-3)
    parser.add_argument('--batch_size',  type=int, default=10)
    parser.add_argument('--repeat',  type=float, default=1)
    parser.add_argument('--loss_train',  type=str, default='mse')
    parser.add_argument('--epochs',  type=int, default=100)

    # PointFit
    parser.add_argument('--fit_deg',  type=int, default=5)
    parser.add_argument('--mask_num',  type=int, default=3000)


    # IO parameters
    parser.add_argument('--output_dir', default='log')
    parser.add_argument('--save_freq', type=int, default=50)

    args = parser.parse_args()
    return args


def getDatasets(args):
    # 读取数据
    n  =  getNorFiled()
    np.random.shuffle(n)
    test_size = int(args.test_size *len(n))

    fileds = np.array([np.concatenate([p,u,v]) for aoa,m,(p,u,v) in n ])

    filedsTest = fileds[:test_size]
    LabelTest = np.array([(aoa,m) for aoa,m,_ in n ])[:test_size]


    filedsTrain = fileds[test_size:]

    class MyDataset:
        def __init__(self,nparray):
            self.data = nparray
        def __getitem__(self, index):
            return self.data[index],self.data[index]
        def __len__(self):
            return len(self.data)
    

    dg_Train = MyDataset(filedsTrain)
    dataset_Train = ds.GeneratorDataset(dg_Train, ["data","label"], shuffle=False)
    dataset_Train = dataset_Train.batch(args.batch_size)
    dataset_Train = dataset_Train.repeat(args.repeat)


    args.input_size = fileds.shape[1]

    return dataset_Train,filedsTest,LabelTest

def trainFP(args):
    fp = PointFit(deg=args.fit_deg)
    log("[PointFit] Fitting to PUV")
    fp.fitPUV()
    saveLinearRegressionPath = args.output_dir+'/PointFit_%ddeg.npz'%args.fit_deg
    fp.saveLinearRegression(path=saveLinearRegressionPath)
    log("[PointFit] saveLinearRegression to ",saveLinearRegressionPath)
    return fp

def train(net,datasetTrain,args):
    # 网络loss
    if 'mse' in args.loss_train.lower():
        loss = nn.MSELoss()

    # 连接前向网络与损失函数
    net_with_loss = nn.WithLossCell(net, loss)
    opt = nn.Adam(net.trainable_params(), args.learning_rate)

    # 定义训练网络，封装网络和优化器
    train_net = nn.TrainOneStepCell(net_with_loss, opt)
    # 设置网络为训练模式
    train_net.set_train()

    # 真正训练迭代过程
    epochs = args.epochs
    avglossLog=[]

    for epoch in range(epochs):
        avgloss = []
        for d in datasetTrain.create_dict_iterator():
            loss_one_branch = train_net(d["data"], d["label"])
            avgloss.append(loss_one_branch)

        avgloss = sum(avgloss)/len(avgloss)
        # save best
        if not avglossLog or avgloss < min(avglossLog):
            log('[train] saving best checkpoint')
            ms.save_checkpoint(net, args.output_dir+"/best.ckpt")
            
        avglossLog.append(avgloss)
        log(f"[train] Epoch: [{epoch} / {epochs}], avg loss: {avgloss}")
        if epoch % args.save_freq == 0:
            log('[train] saving %d checkpoint'%epoch)
            if not os.path.isdir(args.output_dir+"/epochs"): 
                os.makedirs(args.output_dir+"/epochs")
            ms.save_checkpoint(net, args.output_dir+"/epochs/%d.ckpt"%epoch)

    log('[train] saving last checkpoint')
    ms.save_checkpoint(net, args.output_dir+"/last.ckpt")

def test(net,point_fiter : PointFit,datasetTest,LabelTest,args):
    def l1_loss_fun(a,b):
        a=np.array(a).flatten()
        b=np.array(b).flatten()
        return (np.abs(a-b)).mean()
    
    loss_list = []
    for data,label in zip (datasetTest,LabelTest):
        aoa,m = label
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
        l1_loss = l1_loss_fun(data,y_per.asnumpy().flatten())
        log("[test] aoa,m = ",aoa,m, "l1 loss " ,l1_loss )
        loss_list.append(l1_loss)
        
    log("[test] avg l1 loss = " ,sum(loss_list)/len(loss_list) )

if __name__ == '__main__':
    args = get_args()
    log(args)
    
    ms.set_context(mode=ms.PYNATIVE_MODE)

    ms.set_seed(42)
    np.random.seed(42)
    
    if not os.path.isdir(args.output_dir): 
        os.makedirs(args.output_dir)
    # 构造数据集
    datasetTrain,datasetTest,LabelTest = getDatasets(args)
    # 构造网络
    if args.model == 'mlp':
        from model.mlp import MLP
        net = MLP(args)
    elif args.model == 'mlp_s':
        from model.mlp_s import MLP_s
        net = MLP_s(args)
    elif args.model == 'mlp_m':
        from model.mlp_m import MLP_m
        net = MLP_m(args)
    elif args.model == 'mlp_x':
        from model.mlp_x import MLP_x
        net = MLP_x(args)
    else:
        raise "args.model not implement"
    
    log(net.trainable_params())


    if 'train' in args.task :
        # 训练ch模型
        fp = trainFP(args)
        # 训练AE模型
        train(net,datasetTrain,args)

    elif 'test' in args.task :
        fp = PointFit(deg=args.fit_deg)
        if args.point_fit_path:
            fp.laod(args.point_fit_path)
        else:
            fp = trainFP(args)
        
        ms.load_checkpoint(args.model_path,net)

    if 'test' in args.task :
        # 验证模型
        test(net,fp,datasetTest,LabelTest,args)
