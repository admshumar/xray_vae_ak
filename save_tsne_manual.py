import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np


class LatentSpaceTSNE:
    def __init__(self, data, labels, directory, number_of_tnse_components=2):
        if number_of_tnse_components in (2, 3):
            self.number_of_tnse_components = number_of_tnse_components
        else:
            self.number_of_tnse_components = 2
        self.data = data
        self.directory = directory
        self.filename = 'tsne.png'
        self.labels = labels
        self.perplexity = 30
        self.classes = np.unique(labels)
        color_list = ['#00B7BA', '#FFB86F', '#5E6572', '#6B0504', '#BA5C12']
        self.color_list = color_list
        self.color_dict = {i: color_list[i] for i in range(len(color_list))}

        print('the shapes are ', labels.shape, data.shape)

    def save_embedding(self, filename):
        embedded_data = TSNE(n_components=self.number_of_tnse_components,
                             perplexity=self.perplexity).fit_transform(self.data)
        np.save(filename, embedded_data)

    def save_tsne(self):

        embedded_data = TSNE(n_components=self.number_of_tnse_components,
                             perplexity=self.perplexity).fit_transform(self.data)
        fig_3d = plt.figure(dpi=200)

        if self.number_of_tnse_components == 2:
            ax = fig_3d.add_subplot()
            for label in self.classes:
                data = embedded_data[np.where(self.labels == label)]
                ax.scatter(data[:, 0],
                           data[:, 1],
                           c=self.color_dict[label],
                           label=str(label))
            ax.legend(loc='best')
        else:
            ax = fig_3d.add_subplot(projection='3d')
            for label in self.classes:
                data = embedded_data[np.where(self.labels == label)]
                ax.scatter(data[:, 0],
                           data[:, 1],
                           data[:, 2],
                           c=self.color_dict[label],
                           label=str(label))
            ax.legend(loc='best')

        ax.set_title('t-SNE with Perplexity {}'.format(self.perplexity))
        file_path = os.path.join(self.directory, self.filename)
        fig_3d.savefig(file_path)

t = LatentSpaceTSNE(np.load('x_train_latent.npy'), np.load('y_train.npy'), '.')
t.save_embedding('tsne_embedding_x_train.npy')
del t

t = LatentSpaceTSNE(np.load('x_test_latent.npy'), np.load('y_test.npy'), '.')
t.save_embedding('tsne_embedding_x_test.npy')
del t