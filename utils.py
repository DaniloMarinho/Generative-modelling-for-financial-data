import torch
import numpy as np
import os
from scipy import stats
import matplotlib.pyplot as plt
import torch.nn as nn

def D_train(latent_dim, x, G, D, D_optimizer, device, distr="normal", log=None):
    z = sample_from(distr, x.shape[0], latent_dim, device)
    fake_batch = G(z)
    D_scores_on_real = D(x.to(device))
    D_scores_on_fake = D(fake_batch)
    
    # D_loss = -torch.mean(torch.log(D_scores_on_fake) + torch.log(1 - D_scores_on_real))

    y_real, y_fake = torch.ones(x.shape[0], 1).to(device), torch.zeros(x.shape[0], 1).to(device)
    D_real_loss = nn.BCELoss()(D_scores_on_real, y_real)
    D_fake_loss = nn.BCELoss()(D_scores_on_fake, y_fake)
    D_loss = D_real_loss + D_fake_loss

    D_optimizer.zero_grad()
    D_loss.backward()
    D_optimizer.step()

    # log to tensorboard
    if log is not None:
        discriminator_log(log, D_scores_on_real.mean(), D_scores_on_fake.mean(), D_loss)

    return D_loss.data.item()

def G_train(latent_dim, x, G, D, G_optimizer, device, distr="normal", log=None):
    z = sample_from(distr, x.shape[0], latent_dim, device)
    fake_batch = G(z)
    D_scores_on_fake = D(fake_batch)
    
    # G_loss = -torch.mean(torch.log(1 - D_scores_on_fake))

    y = torch.ones(x.shape[0], 1).to(device)
    G_loss = nn.BCELoss()(D_scores_on_fake, y)

    G_optimizer.zero_grad()
    G_loss.backward()
    G_optimizer.step()

    # log to tensorboard
    if log is not None:
        generator_log(log, x, fake_batch, G_loss)

    return G_loss

def sample_from(distr, n_samples, latent_dim, device):
    if distr == "normal":
        return torch.randn(n_samples, latent_dim).to(device)
    elif distr == "exp":
        return torch.Tensor.exponential_(torch.zeros((n_samples, latent_dim)), 2).to(device)
    elif distr == "gamma":
        return torch.distributions.gamma.Gamma(torch.tensor([1.0]), torch.tensor([1.0])).sample((n_samples, latent_dim))[:,:,0].to(device)
    elif distr == "uniform":
        return torch.rand(n_samples, latent_dim).to(device)
    elif distr == "student":
        return torch.distributions.studentT.StudentT(torch.tensor([1.5])).sample((n_samples, latent_dim))[:,:,0].to(device)

def D_wasserstrain(latent_dim, x, G, D, D_optimizer, device, distr="normal", log=None):
    # =======================Train the discriminator=======================#
    D.zero_grad()

    # train discriminator on real
    x_real = x
    x_real = x_real.to(device)

    D_output_real = D(x_real)

    # train discriminator on fake
    z = sample_from(distr, x.shape[0], latent_dim, device)
    x_fake = G(z)

    D_output_fake = D(x_fake)

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

    z = sample_from(distr, x.shape[0], latent_dim, device)

    G_output = G(z)
    D_output = D(G_output)
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

def discriminator_log(log, real_loss, fake_loss, loss):
    writer, idx, ep, num_batches = log
    writer.add_scalar("D_loss/train_step", loss.item(),
                      global_step=ep * num_batches + idx)
    writer.add_scalar("D_loss/real_train_step", real_loss.item(),
                      global_step=ep * num_batches + idx)
    writer.add_scalar("D_loss/fake_train_step", fake_loss.item(),
                      global_step=ep * num_batches + idx)
    
def metrics_log(original_data, fake_data, log, folder="metrics/"):
    writer, idx, ep, num_batches = log
    writer.add_scalar(folder + "ad_distance", AD_distance(original_data, fake_data),
                      global_step=ep * num_batches + idx)
    writer.add_scalar(folder + "absolute_kendall_error", Absolute_Kendall_error(original_data, fake_data),
                      global_step=ep * num_batches + idx)
    
def metrics_log_train(original_data, fake_data, log, folder="metrics/"):
    writer, idx, ep, num_batches = log
    writer.add_scalar(folder + "ad_distance_train", AD_distance(original_data, fake_data),
                      global_step=ep * num_batches + idx)
    writer.add_scalar(folder + "absolute_kendall_error_train", Absolute_Kendall_error(original_data, fake_data),
                      global_step=ep * num_batches + idx)
    
def metrics_log_test(original_data, fake_data, log, folder="metrics/"):
    writer, idx, ep, num_batches = log
    writer.add_scalar(folder + "ad_distance_test", AD_distance(original_data, fake_data),
                      global_step=ep * num_batches + idx)
    writer.add_scalar(folder + "absolute_kendall_error_test", Absolute_Kendall_error(original_data, fake_data),
                      global_step=ep * num_batches + idx)

def make_fake_data(distr, n, latent_dim, G, means, stds):
    z = sample_from(distr, n, latent_dim, "cpu")
    fake_samples = G(z)
    fake_data = fake_samples.cpu().data.numpy()
    fake_data = fake_data * np.array(stds.cpu()) + np.array(means.cpu())
    return fake_data
    
def plot_fake_data(fake_data, log, folder="ecdfs/"):
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

    writer, idx, ep, num_batches = log
    writer.add_figure(folder + "{:04}".format(idx), fig,
                            global_step=ep * num_batches + idx)
    plt.close(fig)

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
    return np.linalg.norm(np.sort(Z_og)-np.sort(Z_gen), 1)/Z_og.shape[0]

def H(z, x):
    return ((1 - z**2) / 2) ** (-x)

###################################################################
# Attempt at Energy Distance model
def ED_model_step(latent_dim, x, G, G_optimizer, device, distr="normal", log=None):

    G.zero_grad()

    z = sample_from(distr, x.shape[0], latent_dim, device)

    G_output = G(z)

    # print(x.device)
    # print(G_output.device)

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
