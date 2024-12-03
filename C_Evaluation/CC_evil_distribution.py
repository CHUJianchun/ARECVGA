
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KernelDensity
from tqdm import trange

from args import args


# Kernel Density Estimation
def distribution(split=30):
    z = np.load('DataB_ProcessedData/All_Mol_latent_code.npy').reshape(-1, args.latent_dim)
    print('Step 1/3')
    for i in trange(z.shape[1]):
        x = z[:, i]
        kde_model = KernelDensity(bandwidth=(x.max()-x.min()) / split, kernel='gaussian').fit(x[:, np.newaxis])
        x_range = np.linspace(x.min() - (x.max()-x.min()) / split * 5, x.max() + (x.max()-x.min()) / split * 5, 500)
        x_log_prob = kde_model.score_samples(x_range[:, np.newaxis])
        x_prob = np.exp(x_log_prob)
        fig = plt.figure(figsize=(10, 10))
        plt.fill_between(x=x_range, y1=x_prob, y2=0, color='blue', alpha=0.5)
        plt.plot(x_range, x_prob, color='gray')
        plt.ylim(0, x_prob.max() * 1.1)
        plt.title(str(i))
        plt.draw()
        plt.savefig('DataE_Figure/EA_KDE/value-pic-{}.png'.format(i))
        plt.pause(1)
        plt.close(fig)
    print('Step 2/3')
    x = np.sum(z**2, axis=1) ** 0.5
    kde_model = KernelDensity(bandwidth=(x.max() - x.min()) / split, kernel='gaussian').fit(x[:, np.newaxis])
    x_range = np.linspace(x.min() - (x.max() - x.min()) / split * 5, x.max() + (x.max() - x.min()) / split * 5, 500)
    x_log_prob = kde_model.score_samples(x_range[:, np.newaxis])
    x_prob = np.exp(x_log_prob)
    fig = plt.figure(figsize=(10, 10))
    plt.fill_between(x=x_range, y1=x_prob, y2=0, color='blue', alpha=0.5)
    plt.plot(x_range, x_prob, color='gray')
    plt.ylim(0, x_prob.max() * 1.1)
    plt.title('Distance')
    plt.draw()
    plt.savefig('DataE_Figure/EA_KDE/distance-pic.png')
    plt.pause(1)
    plt.close(fig)
    print('Step 3/3')
    for i in trange(args.latent_dim):
        x_axis = np.zeros(args.latent_dim)
        x_axis[i] = 1.
        angle = np.zeros(z.shape[0])
        norm_x = np.sqrt(x_axis.dot(x_axis))
        for j in range(z.shape[0]):
            norm_z = np.sqrt(z[j].dot(z[j]))
            dot = x_axis.dot(z[j])
            cosine = dot / (norm_x * norm_z)
            angle[j] = np.arccos(cosine) * 180 / np.pi
        x = angle
        kde_model = KernelDensity(bandwidth=(x.max() - x.min()) / split, kernel='gaussian').fit(x[:, np.newaxis])
        x_range = np.linspace(x.min() - (x.max() - x.min()) / split * 5, x.max() + (x.max() - x.min()) / split * 5, 500)
        x_log_prob = kde_model.score_samples(x_range[:, np.newaxis])
        x_prob = np.exp(x_log_prob)
        fig = plt.figure(figsize=(10, 10))
        plt.fill_between(x=x_range, y1=x_prob, y2=0, color='blue', alpha=0.5)
        plt.plot(x_range, x_prob, color='gray')
        plt.ylim(0, x_prob.max() * 1.1)
        plt.title(str(i))
        plt.draw()
        plt.savefig('DataE_Figure/EA_KDE/angle-pic-{}.png'.format(i))
        plt.pause(1)
        plt.close(fig)


if __name__ == '__main__':
    distribution()
