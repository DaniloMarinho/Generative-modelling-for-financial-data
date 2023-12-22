import pandas as pd
import numpy as np
import torch
import os
from tqdm import tqdm
import argparse
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.utils.data as data_utils



from model import Generator, Discriminator
from utils import D_train, G_train, save_models
from utils import D_wasserstrain, G_wasserstrain, make_fake_data, plot_fake_data
from utils import metrics_log, metrics_log_train, metrics_log_test
from utils import ED_model_step
from utils import AD_distance, Absolute_Kendall_error


from sklearn.model_selection import train_test_split

# changes: can use cpu for local debugging,
#   added version argument for tensorboard,
#   added writer for tensorboard


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Normalizing Flow.')
    parser.add_argument("--epochs", type=int, default=500,
                        help="Number of epochs for training.")
    parser.add_argument("--lr", type=float, default=0.002,
                        help="The learning rate to use for training.")
    parser.add_argument("--batch_size", type=int, default=64, 
                        help="Size of mini-batches for SGD.")
    parser.add_argument("--latent_dim", type=int, default=4, 
                        help="Latent space dimension.")
    parser.add_argument("--g_hidden_dim", type=int, default=64, 
                        help="Impacts generator number of parameters.")
    parser.add_argument("--d_hidden_dim", type=int, default=64, 
                        help="Impacts discriminator number of parameters.")
    parser.add_argument("--latent_distr", type=str, default="normal", 
                        help="Latent space dimension.")
    parser.add_argument("--version", "-v", type=str, default="test",
                        help="Name of run for Tensorboard.")

    args = parser.parse_args()


    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('data', exist_ok=True)

    # Data Pipeline
    print('Dataset loading...')
    # MNIST Dataset
    # transform = transforms.Compose([
    #             transforms.ToTensor(),
    #             transforms.ize(mean=(0.5), std=(0.5))])

    # train_dataset = datasets.MNIST(root='data/MNIST/', train=True, transform=transform, download=True)
    # test_dataset = datasets.MNIST(root='data/MNIST/', train=False, transform=transform, download=False)

    data = pd.read_csv("data/data_train_log_return.csv", names=["idx", "X1", "X2", "X3", "X4"])
    data = data.set_index(["idx"])

    train_data, data_test = train_test_split(data, test_size=0.2)
    
    k = 3  # number of folds
    AD_distances = []
    Kendall_errors = []
    
    # define writer to accompany training
    writer = SummaryWriter(log_dir=f"tb_logs/{args.version}", purge_step=0)

    for i in range(k):
        print(f"{i+1}-th fold:")

        val_data = train_data.iloc[i*len(train_data)//k:(i+1)*len(train_data)//k]
        data_train = pd.concat([train_data.iloc[:i*len(train_data)//k], train_data.iloc[(i+1)*len(train_data)//k:]])

        train = torch.tensor(data_train.values.astype(np.float32))
        test = torch.tensor(val_data.values.astype(np.float32))

        # standardize
        means = train.mean(dim=0, keepdim=True)
        stds = train.std(dim=0, keepdim=True)
        # means = 0 * means
        # stds = 1 + 0 * stds
        train_normalized = (train - means) / stds
        test_normalized = (test - means) / stds

        train_dataset = data_utils.TensorDataset(train_normalized) 
        train_loader = data_utils.DataLoader(dataset = train_dataset, batch_size = args.batch_size, shuffle = True)

        test_dataset = data_utils.TensorDataset(test_normalized) 
        test_loader = data_utils.DataLoader(dataset = test_dataset, batch_size = args.batch_size, shuffle = True)

        print('Dataset Loaded.')

        print('Model Loading...')
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("device", device)

        dim = 4
    
        G = torch.nn.DataParallel(Generator(latent_dim=args.latent_dim, g_hidden_dim=args.g_hidden_dim, g_output_dim=dim)).to(device)
        D = torch.nn.DataParallel(Discriminator(d_input_dim=dim, d_hidden_dim=args.d_hidden_dim)).to(device)

        print('Model loaded.')


        # define loss
        criterion = nn.BCELoss()

        # define optimizers
        G_optimizer = optim.RMSprop(G.parameters(), lr=args.lr)
        D_optimizer = optim.RMSprop(D.parameters(), lr=args.lr, maximize=True)

        print('Start Training :')
    
        n_epoch = args.epochs
        counter = 0
        for epoch in range(1, n_epoch + 1):
            with tqdm(enumerate(train_loader), total=len(train_loader),
                    leave=False, desc=f"Epoch {epoch}") as pbar:
                # for batch_idx, (x, _) in pbar:
                for batch_idx, x in pbar:
                    # print(x)
                    log = (writer, batch_idx, epoch, len(train_loader))

                    # print(batch_idx)

                    # x = x.view(-1, dim)
                    x = x[0]
                    D_wasserstrain(args.latent_dim, x, G, D, D_optimizer, device, args.latent_distr, log=log)
                    if batch_idx % 5 == 0:
                        G_wasserstrain(args.latent_dim, x, G, D, G_optimizer, device, args.latent_distr, log=log)

                    ############################################
                    # ED-GAN:
                    # x=x[0]
                    # ED_model_step(args.latent_dim, x, G, G_optimizer, device, args.latent_distr, log=log)
                    ############################################

                if epoch % 100 == 0:
                    save_models(G, D, 'checkpoints')
                    # plot_fake_data(data.shape[0], args.latent_dim, G, means, stds)

                    fake_data = make_fake_data(args.latent_distr, data.shape[0], args.latent_dim, G, means, stds)
                    print(fake_data)
                    # metrics_log(data, fake_data, log=log)
                    if i==0: plot_fake_data(fake_data, log)

                    fake_data_train = make_fake_data(args.latent_distr, data_train.shape[0], args.latent_dim, G, means, stds)
                    if i < k-1:
                        fake_data_test = make_fake_data(args.latent_distr, data_test.shape[0]//k, args.latent_dim, G, means, stds)
                    else:
                        fake_data_test = make_fake_data(args.latent_distr, data_test.shape[0]//k + data_test.shape[0]%k, args.latent_dim, G, means, stds)
                    # metrics_log_train(data_train, fake_data_train, log=log)
                    if i==0:
                        AD_distances.append(AD_distance(data_train, fake_data_train))
                        Kendall_errors.append(Absolute_Kendall_error(data_train, fake_data_train))
                    else: 
                        AD_distances[counter] += AD_distance(data_train, fake_data_train)
                        Kendall_errors[counter] += Absolute_Kendall_error(data_train, fake_data_train)
                    # metrics_log_test(data_test, fake_data_test, log=log)
                    counter += 1

    print('Training done')

    n = len(AD_distances)
    AD_distance_train = np.array(AD_distances)/n
    absolute_kendall_error_train = np.array(Kendall_errors)/n
    for  i in range(n):
        writer, idx, ep, num_batches = log
        writer.add_scalar("metrics/ad_distance_train", AD_distance_train[i],
                        global_step=ep * num_batches + idx)
        writer.add_scalar("metrics/absolute_kendall_error_train", absolute_kendall_error_train[i],
                        global_step=ep * num_batches + idx)

    # plot_fake_data(data.shape[0], args.latent_dim, G_wasserstrain)
    metrics_log_test(data_test, fake_data_test, log=log)
    writer.close()