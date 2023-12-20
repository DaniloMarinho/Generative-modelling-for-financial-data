import torch
import numpy as np
import os
from scipy import stats
import matplotlib.pyplot as plt

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

def D_wasserstrain(latent_dim, x, G, D, D_optimizer, device, distr="normal", log=None):
    # =======================Train the discriminator=======================#
    D.zero_grad()

    # train discriminator on real
    x_real = x
    x_real = x_real.to(device)

    D_output_real = D(x_real)
    # D_real_score = D_output_real

    # train discriminator on fake
    # z = torch.randn(x.shape[0], latent_dim).to(device)
    z = sample_from(distr, x.shape[0], latent_dim, device)
    x_fake = G(z)

    D_output_fake = D(x_fake)
    # D_fake_score = D_output_fake

    # gradient backprop & optimize ONLY D's parameters
    D_loss = (D_output_real - D_output_fake).mean()
    D_loss.backward()
    D_optimizer.step()

    for p in D.parameters():
        p.data = torch.clamp(p.data, -0.01, 0.01)

    # log to tensorboard
    if log is not None:
        discriminator_log(log, D_output_real.mean(), D_output_fake.mean(), D_loss)

    return D_loss.data.item()


def G_wasserstrain(latent_dim, x, G, D, G_optimizer, device, distr="normal", log=None):
    # =======================Train the generator=======================#
    G.zero_grad()

    # z = torch.randn(x.shape[0], latent_dim).to(device)
    z = sample_from(distr, x.shape[0], latent_dim, device)

    # y = torch.ones(x.shape[0], 1).to(device)

    G_output = G(z)
    D_output = D(G_output)
    #G_loss = criterion(D_output, y)
    G_loss = - D_output.mean()

    # gradient backprop & optimize ONLY G's parameters
    G_loss.backward()
    G_optimizer.step()

    # log to tensorboard
    if log is not None:
        generator_log(log, x, G_output, G_loss)

    return G_loss.data.item()


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
    return np.linalg.norm(np.sort(Z_og)-np.sort(Z_gen), 1)

# def H(u, y):
#     return -torch.log(y)/(torch.log(1 - u**2) - torch.log(2))

def H(z, x):
    return ((1 - z**2) / 2) ** (-x)
