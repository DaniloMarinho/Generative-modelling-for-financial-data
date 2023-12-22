import torch
import numpy as np
import os
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd

# changes: added tensorboard support


def D_train(x, G, D, D_optimizer, criterion, device, log=None):
    #=======================Train the discriminator=======================#
    D.zero_grad()

    # train discriminator on real
    x_real, y_real = x, torch.ones(x.shape[0], 1)
    x_real, y_real = x_real.to(device), y_real.to(device)

    D_output = D(x_real)
    D_real_loss = criterion(D_output, y_real)
    D_real_score = D_output

    # train discriminator on fake
    z = torch.randn(x.shape[0], 100).to(device)
    x_fake, y_fake = G(z), torch.zeros(x.shape[0], 1).to(device)

    D_output = D(x_fake)
    
    D_fake_loss = criterion(D_output, y_fake)
    D_fake_score = D_output

    # gradient backprop & optimize ONLY D's parameters
    D_loss = D_real_loss + D_fake_loss
    D_loss.backward()
    D_optimizer.step()

    # log to tensorboard
    if log is not None:
        discriminator_log(log, D_real_loss, D_fake_loss, D_loss)
        
    return D_loss.data.item()


def G_train(x, G, D, G_optimizer, criterion, device, log=None):
    #=======================Train the generator=======================#
    G.zero_grad()

    z = torch.randn(x.shape[0], 100).to(device)
    y = torch.ones(x.shape[0], 1).to(device)
                 
    G_output = G(z)
    D_output = D(G_output)
    G_loss = criterion(D_output, y)

    # gradient backprop & optimize ONLY G's parameters
    G_loss.backward()
    G_optimizer.step()

    # log to tensorboard
    if log is not None:
        generator_log(log, x, G_output, G_loss)
        
    return G_loss.data.item()

def sample_from(distr, n_samples, latent_dim, device):
    if distr == "normal":
        return torch.randn(n_samples, latent_dim).to(device)
    elif distr == "exp":
        return torch.Tensor.exponential_(torch.zeros((n_samples, latent_dim)), 1).to(device)
    elif distr == "gamma":
        return torch.distributions.gamma.Gamma(torch.tensor([1.0]), torch.tensor([1.0])).sample((n_samples, latent_dim))[:,:,0].to(device)
    elif distr == "uniform":
        return torch.rand(n_samples, latent_dim)

def D_wasserstrain(latent_dim, x, G, D, D_optimizer, device, distr="normal", log=None, a=0.9, b=1.1):
    # =======================Train the discriminator=======================#
    D.zero_grad()

    # train discriminator on real
    x_real = x
    x_real = x_real.to(device)

    # D_real_score = D_output_real

    # train discriminator on fake
    # z = torch.randn(x.shape[0], latent_dim).to(device)
    z = sample_from(distr, x.shape[0], latent_dim, device)
    x_fake = G(z)

    x_fake_std, x_fake_xtr = split_std_xtr(x_fake)
    x_real_std, x_real_xtr = split_std_xtr(x_real)

    D_output_fake_std = D(x_fake_std)
    D_output_fake_xtr = D(x_fake_xtr)

    x_real_std = x_real_std.to(device)
    x_real_xtr = x_real_xtr.to(device)
    
    D_output_real_std = D(x_real_std)
    D_output_real_xtr = D(x_real_xtr)
    D_loss_std = a*(D_output_real_std.mean() - D_output_fake_std.mean())
    D_loss_xtr = b*(D_output_real_xtr.mean() - D_output_fake_xtr.mean())
    # D_fake_score = D_output_fake

    # gradient backprop & optimize ONLY D's parameters
    D_loss = D_loss_std + D_loss_xtr
    D_loss.backward()
    D_optimizer.step()

    for p in D.parameters():
        p.data = torch.clamp(p.data, -0.01, 0.01)

    # log to tensorboard
    # if log is not None:
    #     discriminator_log(log, D_output_real.mean(), D_output_fake.mean(), D_loss)

    return D_loss.data.item()

def split_std_xtr(x, alpha=0.95):
    quantiles1 = torch.quantile(x[:,0], alpha, interpolation="lower")
    quantiles2 = torch.quantile(x[:,1], alpha, interpolation="lower")
    quantiles3 = torch.quantile(x[:,2], alpha, interpolation="lower")
    quantiles4 = torch.quantile(x[:,3], alpha, interpolation="lower")
    data_extreme = x[(x[:,0]>=quantiles1) | (x[:,1]>=quantiles2) | (x[:,2]>=quantiles3) | (x[:,3]>=quantiles4), :]
    data_std = x[~((x[:,0]>=quantiles1) | (x[:, 1]>=quantiles2) | (x[:,2]>=quantiles3) | (x[:,3]>=quantiles4)), :]
    return torch.tensor(data_std), torch.tensor(data_extreme)


def G_wasserstrain(latent_dim, x, G, D, G_optimizer, device, distr="normal", log=None, a=0.9, b=1.1):
    # =======================Train the generator=======================#
    G.zero_grad()

    # z = torch.randn(x.shape[0], latent_dim).to(device)
    z = sample_from(distr, x.shape[0], latent_dim, device)

    # y = torch.ones(x.shape[0], 1).to(device)

    G_output = G(z)
    G_output_std, G_output_xtr = split_std_xtr(G_output)
    D_output_std = D(G_output_std)
    D_output_xtr = D(G_output_xtr)
    #G_loss = criterion(D_output, y)
    G_loss = - a*D_output_std.mean() - b*D_output_xtr.mean()

    # gradient backprop & optimize ONLY G's parameters
    G_loss.backward()
    G_optimizer.step()

    # log to tensorboard
    if log is not None:
        generator_log(log, x, G_output, G_loss)

    return G_loss.data.item()

###################################################################
# Attempt at Energy Distance model
def ED_model_step(latent_dim, x, G, G_optimizer, device, distr="normal", log=None):

    G.zero_grad()

    z = sample_from(distr, x.shape[0], latent_dim, device)
    
    G_output = G(z)
    
    #G_loss=energy_distance_same_length(x, G_output)
    G_loss1 = energy_distance1(x, G_output)
    #G_loss2 = energy_distance2(x, G_output)
#####################################
    # x_reshaped = x.view((x.shape[0], 1, x.shape[1]))
    # out_reshaped = G_output.view((1, G_output.shape[0], G_output.shape[1]))

    # normsA = torch.linalg.norm(x_reshaped - out_reshaped, axis=2) 
    # A = torch.sum(normsA)
    # normsB = torch.linalg.norm(x_reshaped - x_reshaped, axis=2) 
    # B = torch.sum(normsB)
    # normsC = torch.linalg.norm(out_reshaped - out_reshaped, axis=2) 
    # C = torch.sum(normsC)

    # n=x.shape[0]
    # m=G_output.shape[0]
    # G_loss = 2*A/(n*m) - C/(m**2)  - B/(n**2) 
#####################################
    #verify(G, G_loss1, G_loss2)

    G_loss1.backward()

    G_optimizer.step()

    if log is not None:
        generator_log(log, x, G_output, G_loss1)

    return G_loss1.data.item()

def verify(G, l1, l2):

    if torch.abs(l1-l2)>10**(-6): 
        print("Losses don't match")
        print(f"{l1} vs {l2}")

    l1.backward(retain_graph=True)

    A=[]

    for n ,p in G.named_parameters():
        A.append(p.grad)
        
    G.zero_grad()

    l2.backward()

    B=[]

    for p in G.parameters():
        B.append(p.grad)

    for i in range(len(A)):
        equal=True
        if np.linalg.norm(A[i] - B[i], np.inf) > 10**(-6):
            equal = False

    if (not equal): print("Gradients don't match")

    return    

def energy_distance2(x, y):
    n = x.shape[0]
    m = y.shape[0]

    A = torch.tensor([0.], requires_grad=True)
    for i in range(n):
        for j in range(m):
            A = A + torch.linalg.norm(x[i,:]-y[j,:])

    # print(f"A for l2 {A}")
    B = torch.tensor([0.], requires_grad=True)
    for i in range(n):
        for j in range(n):
            B = B + torch.linalg.norm(x[i,:] - x[j,:])
    # print(f"B for l2 {B}")

    C = torch.tensor([0.], requires_grad=True)
    for i in range(m):
        for j in range(m):
            C = C + torch.linalg.norm(y[i,:] - y[j,:])
    # print(f"C for l2 {C}")

    result = 2*A/(n*m) - C/(m**2)  - B/(n**2) 
    return result


def energy_distance1(x, y):
    n=x.shape[0]
    m=y.shape[0]

    # A = torch.tensor([0.], requires_grad=True)
    # for i in range(n):
    #     for j in range(m):
    #         A = A + torch.linalg.norm(x[i,:]-y[j,:])
    # B = torch.tensor([0.], requires_grad=True)
    # for i in range(n):
    #     for j in range(n):
    #         B = B + torch.linalg.norm(x[i,:] - x[j,:])
    # C = torch.tensor([0.], requires_grad=True)
    # for i in range(m):
    #     for j in range(m):
    #         C = C + torch.linalg.norm(y[i,:] - y[j,:])

    # x_reshaped = torch.tensor(x[:, None, :], requires_grad=True)
    # y_reshaped = torch.tensor(y[None, :, :], requires_grad=True)
    x_reshaped1 = torch.unsqueeze(x, 1)
    y_reshaped2 = torch.unsqueeze(y, 0)
    x_reshaped2 = torch.unsqueeze(x, 0)
    y_reshaped1 = torch.unsqueeze(y, 1)
    # x_reshaped = torch.reshape(x, (x.shape[0], 1, x.shape[1]))
    # y_reshaped = torch.reshape(y, (1, y.shape[0], y.shape[1]))

    # x_reshaped = x.view((x.shape[0], 1, x.shape[1]))
    # y_reshaped = y.view((1, y.shape[0], y.shape[1])) 
    
    normsA = torch.linalg.norm(x_reshaped1 - y_reshaped2, axis=2) 
    A = torch.sum(normsA)
    normsB = torch.linalg.norm(x_reshaped1 - x_reshaped2, axis=2) 
    B = torch.sum(normsB) 
    normsC = torch.linalg.norm(y_reshaped1 - y_reshaped2, axis=2) 
    C = torch.sum(normsC)

    # print(f"\nA for l1 {A}")
    # print(f"B for l1 {B}")
    # print(f"C for l1 {C}")


    result = 2*A/(n*m) - C/(m**2)  - B/(n**2) 
    return result
###################################################################


def save_models(G, D, folder):
    torch.save(G.state_dict(), os.path.join(folder,'G.pth'))
    torch.save(D.state_dict(), os.path.join(folder,'D.pth'))


def load_model(G, folder):
    ckpt = torch.load(os.path.join(folder,'G.pth'))
    G.load_state_dict({k.replace('module.', ''): v for k, v in ckpt.items()})
    return G

def generator_log(log, real_batch, fake_batch, loss):
    writer, idx, ep, num_batches = log
    writer.add_scalar("G_loss/train_step", loss.item(),
                      global_step=ep * num_batches + idx)

    # if idx % 50 == 0:
    #     for i in range(0, real_batch.shape[0], 8):
    #         img = fake_batch[i].reshape(28, 28).cpu().detach().numpy()
    #         fig, ax = plt.subplots(1, 1)
    #         ax.imshow(img, cmap="gray", vmin=-1, vmax=1)
    #         ax.axis("off")
    #         fig.tight_layout()
    #         writer.add_figure("samples/{:04}".format(i), fig,
    #                           global_step=ep * num_batches + idx)
    #         plt.close(fig)

    #     ref = real_batch[0].reshape(28, 28).cpu().detach().numpy()
    #     fig, ax = plt.subplots(1, 1)
    #     ax.imshow(ref, cmap="gray", vmin=-1, vmax=1)
    #     ax.axis("off")
    #     fig.tight_layout()
    #     writer.add_figure("reference", fig,
    #                       global_step=ep * num_batches + idx)
    #     plt.close(fig)

def discriminator_log(log, real_loss, fake_loss, loss):
    writer, idx, ep, num_batches = log
    writer.add_scalar("D_loss/train_step", loss.item(),
                      global_step=ep * num_batches + idx)
    writer.add_scalar("D_loss/real_train_step", real_loss.item(),
                      global_step=ep * num_batches + idx)
    writer.add_scalar("D_loss/fake_train_step", fake_loss.item(),
                      global_step=ep * num_batches + idx)
    
# def plot_fake_data(n, latent_dim, G, means, stds):
#     z = torch.empty(n,latent_dim).normal_()
#     fake_samples = G(z)
#     fake_data = fake_samples.cpu().data.numpy()
#     fake_data = fake_data * np.array(stds) + np.array(means)

#     ecdf1 = stats.ecdf(fake_data[:,0])
#     ecdf2 = stats.ecdf(fake_data[:,1])
#     ecdf3 = stats.ecdf(fake_data[:,2])
#     ecdf4 = stats.ecdf(fake_data[:,3])

#     fig, ax = plt.subplots(figsize = (12,6))
#     ecdf1.cdf.plot(ax)
#     ecdf2.cdf.plot(ax)
#     ecdf3.cdf.plot(ax)
#     ecdf4.cdf.plot(ax)
#     ax.set_xlabel('Value')
#     ax.set_ylabel('Empirical CDF')
#     plt.legend(["X1", "X2", "X3", "X4"])
#     plt.show()
    
def metrics_log(original_data, fake_data, log):
    writer, idx, ep, num_batches = log
    writer.add_scalar("metrics/ad_distance", AD_distance(original_data, fake_data),
                      global_step=ep * num_batches + idx)
    writer.add_scalar("metrics/absolute_kendall_error", Absolute_Kendall_error(original_data, fake_data),
                      global_step=ep * num_batches + idx)
    
def metrics_log_train(original_data, fake_data, log):
    writer, idx, ep, num_batches = log
    writer.add_scalar("metrics/ad_distance_train", AD_distance(original_data, fake_data),
                      global_step=ep * num_batches + idx)
    writer.add_scalar("metrics/absolute_kendall_error_train", Absolute_Kendall_error(original_data, fake_data),
                      global_step=ep * num_batches + idx)
    
def metrics_log_test(original_data, fake_data, log):
    writer, idx, ep, num_batches = log
    writer.add_scalar("metrics/ad_distance_test", AD_distance(original_data, fake_data),
                      global_step=ep * num_batches + idx)
    writer.add_scalar("metrics/absolute_kendall_error_test", Absolute_Kendall_error(original_data, fake_data),
                      global_step=ep * num_batches + idx)

def make_fake_data(distr, n, latent_dim, G, means, stds):
    # z = torch.empty(n,latent_dim).normal_()
    z = sample_from(distr, n, latent_dim, "cpu")
    fake_samples = G(z)
    fake_data = fake_samples.cpu().data.numpy()
    fake_data = fake_data * np.array(stds) + np.array(means)
    return fake_data
    
def plot_fake_data(fake_data, log):
    ecdf1 = stats.ecdf(fake_data[:,0])
    ecdf2 = stats.ecdf(fake_data[:,1])
    ecdf3 = stats.ecdf(fake_data[:,2])
    ecdf4 = stats.ecdf(fake_data[:,3])

    fig, ax = plt.subplots(figsize = (12,6))
    ecdf1.cdf.plot(ax)
    ecdf2.cdf.plot(ax)
    ecdf3.cdf.plot(ax)
    ecdf4.cdf.plot(ax)
    ax.set_xlabel('Value')
    ax.set_ylabel('Empirical CDF')
    ax.legend(["X1", "X2", "X3", "X4"])
    # plt.show()

    writer, idx, ep, num_batches = log
    writer.add_figure("ecdfs/{:04}".format(idx), fig,
                            global_step=ep * num_batches + idx)
    plt.close(fig)

    import numpy as np

def mod_prob(v_og, v_gen, n):
    v_gen=np.sort(v_gen)
    u_tilde=np.array([sum(v_og <= v_gen[i]) for i in range(n)])
    return (u_tilde+1)/(n+2)

# def AD_distance(X_og, X_gen):
#     X_og=np.array(X_og)
#     X_gen=np.array(X_gen)
#     n=X_og.shape[0]
#     d=X_og.shape[1]

#     u_tilde=np.array([mod_prob(X_og[:,i], X_gen[:,i], n) for i in range(d)])

#     weigth=np.array([(2*i + 1) for i in range(n)])
#     W = - n - np.mean( ( weigth  * ( np.log(u_tilde) + np.log(1 - u_tilde[::-1]) ) ), axis=0 )

#     return np.mean(W)

def AD_distance(X_og, X_gen):
    X_og=np.array(X_og)
    X_gen=np.array(X_gen)
    n=X_og.shape[0]
    d=X_og.shape[1]

    u_tilde=np.array([mod_prob(X_og[:,i], X_gen[:,i], n) for i in range(d)])

    weigth=np.array([(2*i + 1) for i in range(n)])
    W = - n - np.mean( ( weigth  * ( np.log(u_tilde) + np.log(1 - u_tilde[:, ::-1]) ) ), axis=1)

    return np.mean(W)

def pseudo_obs(x):
    n=x.shape[0]
    d=x.shape[1]
    M=np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if(j==i):
                M[i,i] = False
            else:
                bool=True
                for c in range(d):
                    if x[j,c] >= x[i,c]: 
                        bool=False
                M[i,j]=bool

    return np.sum(M, 0)/(n-1)

def Absolute_Kendall_error(X_og, X_gen):
    Z_og=pseudo_obs(np.array(X_og))
    Z_gen=pseudo_obs(np.array(X_gen))
    return np.linalg.norm(np.sort(Z_og)-np.sort(Z_gen), 1)/X_og.shape[0]

# def H(u, y):
#     return -torch.log(y)/(torch.log(1 - u**2) - torch.log(2))

def H(z, x):
    return ((1 - z**2) / 2) ** (-x)
