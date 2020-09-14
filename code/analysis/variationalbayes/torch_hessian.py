import torch
from torch.autograd import grad


def hessian(loss, params):
    loss_grad = grad(loss, params, create_graph=True)
    cnt = 0
    for g in loss_grad:
        g_vector = g.contiguous().view(-1) if cnt == 0 else torch.cat([g_vector, g.contiguous().view(-1)])
        cnt = 1
    l = g_vector.size(0)
    h = []
    for idx in range(l):
        grad2rd = grad(g_vector[idx], params, create_graph=True)
        cnt = 0
        for g in grad2rd:
            g2 = g.contiguous().view(-1) if cnt == 0 else torch.cat([g2, g.contiguous().view(-1)])
            cnt = 1
        h.append(g2)
    return torch.stack(h) #.cpu().data.numpy()
    
    
def function_hessian(y, x):
    # compute hessian
    x_1grad = grad(y, x, create_graph=True)[0]
    print("x_1grad:", x_1grad)
    h = []
    for i in range(x_1grad.size(0)):
        x_2grad = grad(x_1grad[i], x, create_graph=True)[0]
        h.append(x_2grad)

    h = torch.stack(h)
    return h

a = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
b = torch.tensor([4.0, 5.0, 6.0], requires_grad=True)
print("a:", a)
print("b:", b)
y = torch.sum(a*a + a[:, None]*b[None, :])
print("y:", y)    
hy = hessian(y, [a, b])
print("hy:", hy)


