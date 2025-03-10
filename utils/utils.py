import torch
import copy

class AverageMeter(object):
    """
    Computes and stores the average and
    current value.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def Accuracy(y,y_predict):
    leng = len(y)
    miss = 0
    for i in range(leng):
        if not y[i]==y_predict[i]:
            miss +=1
    return (leng-miss)/leng


def soft_predict(Z,temp):
    m,n = Z.shape
    Q = torch.zeros(m,n)
    Z_sum = torch.sum(torch.exp(Z/temp),dim=1)
    for i in range(n):
        Q[:,i] = torch.exp(Z[:,i]/temp)/Z_sum
    return Q

def average_weights(w):
    """
    average the weights from all local models
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

def accuracy(output, target, topk=(1,)):
       """Computes the precision@k for the specified values of k"""
       maxk = max(topk)
       batch_size = target.size(0)

       _, pred = output.topk(maxk, 1, True, True)
       pred = pred.t()
       correct = pred.eq(target.view(1, -1).expand_as(pred))

       res = []
       for k in topk:
           correct_k = correct[:k].view(-1).float().sum(0)
           res.append(correct_k.mul_(100.0 / batch_size))
       return res