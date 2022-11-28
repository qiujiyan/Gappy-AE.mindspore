import numpy as np


class GappyPOD():
    def __init__(self, x):
        if len(x.shape) != 2:
            raise Exception('error GappyPOD len(input.shape)!=2 ')
        self.x = x

    def train(self):
        self.u, self.s, self.vh = np.linalg.svd(self.x, full_matrices=False)
        self.num_mod = len(self.s)
        return self.u, self.s, self.vh

    def test(self, x, mask, num_mod=-1):
        if num_mod == -1:
            num_mod = self.num_mod
        # assert(x.shape == mask.shape)
        assert(len(x.shape) == 1)
        assert(num_mod <= self.num_mod)

        u, s, vh = self.__cutOffSVD__(num_mod)
        mask = np.array(mask, 'int64')
        vh_mask = vh[:, mask]
        gcross = vh_mask@vh_mask.T
        km = np.linalg.cond(gcross)
        f = vh_mask@x[mask]
        atuta = np.linalg.solve(gcross, f)
        ututa = atuta@vh
        return ututa, km

    def test_resDic(self, x, mask, num_mod=-1):
        if num_mod == -1:
            num_mod = self.num_mod
        # assert(x.shape == mask.shape)
        assert(len(x.shape) == 1)
        assert(num_mod <= self.num_mod)

        u, s, vh = self.__cutOffSVD__(num_mod)
        mask = np.array(mask, 'int64')
        vh_mask = vh[:, mask]
        gcross = vh_mask@vh_mask.T
        km = np.linalg.cond(gcross)
        f = vh_mask@x[mask]
        atuta = np.linalg.solve(gcross, f)
        ututa = atuta@vh
        result = {
            "ututa":ututa,
            "atuta":atuta,
            "km":km,
            "gcross":gcross,
            "mask":mask,
            "f":f,
        }
        return result

    def __km_dic__(self,mask):
        mask = np.array(mask, 'int64')
        km_dic = {}
        for num_mod in range(1,self.num_mod):
            u, s, vh = self.__cutOffSVD__(num_mod)
            vh_mask = vh[:, mask]
            gcross = vh_mask@vh_mask.T
            km = np.linalg.cond(gcross)
            km_list[num_mod] = km
        return km_dic

    def __cutOffSVD__(self, num_mod):
        self.cu = self.u[:, :num_mod]
        self.cs = self.s[:num_mod]
        self.cvh = self.vh[:num_mod, :]
        return self.cu, self.cs, self.cvh

    def __gappySVD__(self):
        self.gu, self.gs, self.gvh = self.u, self.s, self.vh[:, self.index]
        return self.gu, self.gs, self.gvh