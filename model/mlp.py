
import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import grad

class MLP(nn.Cell):
    """
    LeNet-5网络结构
    """
    def __init__(self,args ):
        self.input_size = args.input_size
        self.mod_number = args.mod_number
        
        super(MLP, self).__init__()
        self.f1=nn.Dense(self.input_size , 6000)
        self.f2=nn.Dense(6000, 1280)
        self.f3=nn.Dense(1280, self.mod_number)
        self.f4=nn.Dense(self.mod_number,1280)
        self.f5=nn.Dense(1280, 6000)
        self.f6=nn.Dense(6000, self.input_size )
        self.layers = [
            self.f1,
            self.f2,
            self.f3,
            self.f4,
            self.f5,
        ]
        self.relu = nn.ReLU()

        self.pod_loss = nn.MSELoss()
        
    def encoder(self,x):
        x = self.f1(x)
        x = self.relu(x)
        x = self.f2(x)
        x = self.relu(x)
        x = self.f3(x)
        x = self.relu(x)
        return x

    def decoder(self,x):
        x = self.f4(x)
        x = self.relu(x)
        x = self.f5(x)
        x = self.relu(x)
        x = self.f6(x)
        return x
        
    def decoder_loss(self, code, y):
        per = self.decoder(code) 
        loss = self.pod_loss(per,y)
        return loss

    def gappy_decoder_loss(self, code, y ,mask_map):
        per = self.decoder(code)
        per = mask_map*per
        y = mask_map*y
        loss = self.pod_loss(per,y)
        return loss

    def loss_decoder_helper(self,fix_y,mask_map):
        def loss_decoder(input_x):
            if len(input_x.shape)!=2:    input_x =  Tensor(input_x.astype(np.float32)).reshape((1,input_x.shape[0]))
            
            if len(mask_map.shape)!=2: 
                mask_map_t = Tensor(mask_map.astype(np.float32)).reshape((1,mask_map.shape[0]))
            else:
                mask_map_t = mask_map
            return self.gappy_decoder_loss(input_x,fix_y,mask_map_t).asnumpy()
        return loss_decoder


    def grad_loss_decoder_helper(self,y,mask_map):
        def fp(input):
            def fn(x,y,m):
                return self.gappy_decoder_loss(x,y,m)
            if len(input.shape)!=2: 
                input = Tensor(input.astype(np.float32)).reshape((1,input.shape[0]))

            if len(mask_map.shape)!=2: 
                mask_map_t =  Tensor(mask_map.astype(np.float32)).reshape((1,mask_map.shape[0]))
            else:
                mask_map_t = mask_map
            gradient = grad(fn, 0, None)(input,y,mask_map_t)
            return gradient.asnumpy().flatten()
        return fp
        
    def construct(self, x):
        for layer in self.layers:
            x = layer(x)
            x = self.relu(x)
        x = self.f6(x)
        return x
