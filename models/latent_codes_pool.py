import random
import torch
class LatentCodesPool:
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:  
            self.num_ws = 0
            self.ws = []
    def query(self, ws):
        if self.pool_size == 0:  
            return ws
        return_ws = []
        for w in ws:  
            if w.ndim == 2:
                i = random.randint(0, len(w) - 1)  
                w = w[i]
            self.handle_w(w, return_ws)
        return_ws = torch.stack(return_ws, 0)   
        return return_ws
    def handle_w(self, w, return_ws):
        if self.num_ws < self.pool_size:  
            self.num_ws = self.num_ws + 1
            self.ws.append(w)
            return_ws.append(w)
        else:
            p = random.uniform(0, 1)
            if p > 0.5:  
                random_id = random.randint(0, self.pool_size - 1)  
                tmp = self.ws[random_id].clone()
                self.ws[random_id] = w
                return_ws.append(tmp)
            else:  
                return_ws.append(w)
