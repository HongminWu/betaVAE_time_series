# %%
import numpy as np
import matplotlib.pylab as plt
import random
import pandas as pd
import ipdb
# %%
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable
from torch.distributions import normal
import torch.utils.data as utils

# %%
from torch.optim import lr_scheduler

# %%
"""
# Loading data
"""

# %%
# Xc = np.load('./data/control_beats.npy')
# Xp = np.load('./data/positive_beats.npy')

# X = np.concatenate((Xc, Xp))

X = np.load('./data/nsamples.npy')

figure_path = './log/figures'

Y = np.zeros(len(X))

# %%
print(X.shape)

# %%
def remap(x, out_min, out_max):
    in_min, in_max = np.min(x), np.max(x)
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

# %%
X = np.array([remap(x, -1, 1) for x in X])

# %%
plt.figure()
for x in X[:20]:
    plt.plot(x)
plt.title('original 20 samples')
plt.savefig(figure_path + '/samples.png',format='png')

# %%

train = torch.utils.data.TensorDataset(torch.Tensor(X), torch.Tensor(Y))
train_loader = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True) 

# %%
"""
# Neural network
"""

# %%
nn_dim = 50
nn_dim2 = 25
latent_dim = 10
img_shape = len(X[0])

# %%
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(img_shape, nn_dim)
        self.fc2 = nn.Linear(nn_dim, nn_dim2)
        
        self.fc21 = nn.Linear(nn_dim2, latent_dim)
        self.fc22 = nn.Linear(nn_dim2, latent_dim)
        
        self.fc3 = nn.Linear(latent_dim, nn_dim2)
        self.fc4 = nn.Linear(nn_dim2, nn_dim)
        self.fc5 = nn.Linear(nn_dim, img_shape)

    def encode(self, x):
        h1 = F.leaky_relu(self.fc1(x))
        h2 = F.leaky_relu(self.fc2(h1))
        return self.fc21(h2), self.fc22(h2)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
    
    def reparameterize_static(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        return std.add_(mu)

    def decode(self, z):
        h3 = F.leaky_relu(self.fc3(z))
        h4 = F.leaky_relu(self.fc4(h3))
        return self.fc5(h4)
#         return torch.tanh(self.fc5(h4))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, img_shape))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# %%
def visualize(model, latent_dim, rangee, epoch):
    viz = []
    
    random_index = np.random.randint(len(X)-1)
    X_random = X[random_index]
    zi, zi_std = model.encode(torch.Tensor(X_random))
    
#     zi = torch.randn(latent_dim)
    
    for j in range(latent_dim):    
        buf = []   
        for i in rangee:
            x = zi.clone()
            x[j] += i
            zi2 = torch.zeros(latent_dim)
            gen_imgs = model.decode(model.reparameterize_static(x, zi2))
            gen_imgs_np = gen_imgs.data.numpy()
            buf.append(gen_imgs_np)
        viz.append(buf)
    
    N, M = 5, int(latent_dim / 5) # latent_dim = M x N. 
    fig, axs = plt.subplots(M, N, sharex=True, sharey=True, figsize=(int(N*4), int(M*4)))
    lw = 2
    
    k = 0
    for i in range(M): # M: rows
        for j in range(N): # N: columns
            axs[i, j].plot(X_random, ls = '--', color = 'black', label='sample')
            ts = np.array(viz[k]).T
            axs[i, j].plot(ts[:, 0], lw = lw, color = 'red',   label='z_%s=%s'%(k, rangee[0]))
            axs[i, j].plot(ts[:, 1], lw = lw, color = 'green', label='z_%s=%s'%(k, rangee[1]))
            axs[i, j].plot(ts[:, 2], lw = lw, color = 'blue',  label='z_%s=%s'%(k, rangee[2]))
            axs[i, j].set_title('latent: z_%s'%k)
            axs[i, j].legend(bbox_to_anchor=(0, 0.02), loc=3, borderaxespad=0.2, ncol=2)
            k += 1
    fig.suptitle('Training: epoch_%s with total %s latent states'%(epoch,latent_dim))
    fig.savefig(figure_path + '/epoch_%s.png'%epoch, format="png")
    # plt.show()
    

def visualize_beat_reconstruction(X, model):
    I = torch.randint(len(X)-1, (1, ))
    x_input = torch.Tensor(np.array([X[I]]))
    x_vector = model.encode(x_input)
    x_reconstruct = model.decode(model.reparameterize(x_vector[0], x_vector[1]))

    plt.figure()
    plt.title('Random reconstructed beat')
    plt.plot(X[I], label='sample')
    plt.plot(x_reconstruct.data.numpy()[0], label = 'reconstructed')
    plt.legend()
    plt.savefig(figure_path + '/random_reconstructed_beat.png')

# %%
"""
# Training
"""

# %%
def loss_function(recon_x, x, mu, logvar):
    BCE = F.mse_loss(recon_x, x.view(-1, img_shape), reduction='sum')
#     BCE = F.binary_cross_entropy(recon_x, x.view(-1, img_shape), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE, KLD

# %%
epochs = 100

beta = 100

C, C_final = 0, 25
C_stop = epochs #25
C_delta = (1 / C_stop) * C_final

# %%
model = VAE()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# %%
list_of_BCE = []
list_of_KL  = []
list_of_C   = []
list_of_KLC_beta = []
list_of_loss = []
for epoch in range(epochs):
    model.train()
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data
        optimizer.zero_grad()
        
        recon_batch, mu, logvar = model(data)
        
        try:
            BCE, KLD = loss_function(recon_batch, data, mu, logvar)
            BCE, KLD = BCE / len(data), KLD / len(data)
            
            loss = BCE + beta * torch.abs(KLD - C)
            
            loss.backward()
            optimizer.step()
        except:
            continue
            
        if epoch > C_stop:
            C = C_final

#     if epoch % 10 == 0:                
    print('Epoch', str(epoch), 
          '| BCE', BCE.data.numpy(), 
          '| KL', KLD.data.numpy(), 
          '| C', C, 
          '| (KL-C) * beta', torch.abs(KLD - C).data.numpy() * beta, 
          '| Loss', loss.data.numpy())

    list_of_BCE.append(BCE.data.numpy())
    list_of_KL.append(KLD.data.numpy())
    list_of_C.append(C)
    list_of_KLC_beta.append(torch.abs(KLD - C).data.numpy() * beta)
    list_of_loss.append(loss.data.numpy())
    
    visualize(model, latent_dim, np.arange(-3, 3.5, 1.5), epoch)

    C += C_delta

# %%
viz = []

random_index = np.random.randint(len(X)-1)
X_random = X[random_index]
zi, zi_std = model.encode(torch.Tensor(X_random))
# zi = torch.randn(latent_dim)
rangee = np.arange(-3, 4, 1.5)
for j in range(latent_dim):    
    buf = []   
    for i in rangee:
        x = zi.clone()
        x[j] += i
        zi2 = torch.zeros(latent_dim)
        gen_imgs = model.decode(model.reparameterize_static(x, zi2))
        gen_imgs_np = gen_imgs.data.numpy()
        buf.append(gen_imgs_np)
    viz.append(buf)

N, M = 5, int(latent_dim / 5)
fig, axs = plt.subplots(M, N, sharex=True, sharey=True, figsize=(int(N*4), int(M*4)))
lw = 2

k = 0
for i in range(M): # rows
    for j in range(N): # columns
        axs[i, j].plot(X_random, ls = '--', color = 'black', label='random sample')
        
        ts = np.array(viz[k]).T
        axs[i, j].plot(ts[:, 0], lw = lw, color = 'red',    label='z_%s=%s'%(k, rangee[0]))
        axs[i, j].plot(ts[:, 1], lw = lw, color = 'green',  label='z_%s=%s'%(k, rangee[1]))
        axs[i, j].plot(ts[:, 2], lw = lw, color = 'blue',   label='z_%s=%s'%(k, rangee[2]))
        axs[i, j].plot(ts[:, 3], lw = lw, color = 'gold',   label='z_%s=%s'%(k, rangee[3]))
        axs[i, j].plot(ts[:, 4], lw = lw, color = 'grey',   label='z_%s=%s'%(k, rangee[4]))
        axs[i, j].set_title('latent: z_%s'%k)
        axs[i, j].legend(bbox_to_anchor=(0, 0.02), loc=3, borderaxespad=0.2, ncol=2)        
        k += 1
fig.suptitle("Trained with total of %s latent states"%latent_dim)        
fig.savefig(figure_path + '/result.png', format='png')
# plt.show()

visualize_beat_reconstruction(X, model)

# plot the training process
# list_of_KL  = []
# list_of_C   = []
# list_of_KLC_beta = []
# list_of_loss = []

fig, ax = plt.subplots()
ax.plot(list_of_BCE, label='BCE')
plt.legend()
fig.savefig(figure_path + '/BCE.png', format='png')

fig, ax = plt.subplots()
ax.plot(list_of_KL, label='KLD')
plt.legend()
fig.savefig(figure_path + '/KLD.png', format='png')

fig, ax = plt.subplots()
ax.plot(list_of_C, label='C')
plt.legend()
fig.savefig(figure_path + '/C.png', format='png')

fig, ax = plt.subplots()
ax.plot(list_of_KLC_beta, label='KLC_beta')
plt.legend()
fig.savefig(figure_path + '/KLC_beta.png', format='png')

fig, ax = plt.subplots()
ax.plot(list_of_loss, label='loss')
plt.legend()
fig.savefig(figure_path + '/loss.png', format='png')

