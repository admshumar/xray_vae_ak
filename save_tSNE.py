import os
from sklearn.manifold import TSNE
import matplotlib
import matplotlib.colors
import matplotlib.pyplot as plt


class LatentSpaceTSNE:
    def __init__(self, data, labels, filename, directory, number_of_tnse_components=2):
        if number_of_tnse_components in (2, 3):
            self.number_of_tnse_components = number_of_tnse_components
        else:
            self.number_of_tnse_components = 2
        self.data = data
        self.directory = directory
        self.filename = filename
        self.labels = labels
        self.classes = np.unique()
        self.perplexity_list = [30]
        self.color_list = ['#00B7BA', '#FFB86F', '#5E6572', '#6B0504', '#BA5C12']

        print('the shapes are ',labels.shape, data.shape)

    def save_tsne(self):

        for perplexity in self.perplexity_list:
            embedded_data = TSNE(n_components=self.number_of_tnse_components,
                                 perplexity=perplexity).fit_transform(self.data)
            fig_3d = plt.figure(dpi=200)


        if self.number_of_tnse_components == 2:
            ax = fig_3d.add_subplot()
            ax.scatter(embedded_data[:, 0],
                       embedded_data[:, 1],
                       c=self.labels,
                       cmap=matplotlib.colors.ListedColormap(self.color_list))
            ax.legend(('1', '2', '3'))
            #plt.gca().legend()
        else:
            ax = fig_3d.add_subplot(projection='3d')
            ax.scatter(embedded_data[:, 0],
                       embedded_data[:, 1],
                       embedded_data[:, 2],
                       c=self.labels,
                       cmap=matplotlib.colors.ListedColormap(self.color_list))
            ax.legend(('1', '2', '3'))

        ax.set_title('t-SNE with Perplexity {}'.format(perplexity))
        filename = 'latent_tsne_{}.png'.format(perplexity)
        file_path = os.path.join(self.directory, self.filename)
        fig_3d.savefig(file_path)
