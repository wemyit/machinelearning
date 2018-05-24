import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.cluster as cluster
import sklearn.decomposition as decomposition
import time
from IPython.core.display import display, HTML
from sklearn import metrics

data = pd.read_csv("data/Final-Data-Set.csv")


def print_table(header, data):
    display(
        HTML(
            '''
            <style>
                .rendered_html table: {{
                    width: 100%;
                    text-align:center;
                }}
                .rendered_html td,
                .rendered_html th {{
                    text-align: center;
                }}
            </style>
            <table border="1">
                <thead>
                    {}
                </thead>
                <tbody>
                    <tr>
                        {}
                    </tr>
                </tbody>
            <table>
            '''.format(header, '</tr><tr>'.join(
                '<td>{}</td>'.format(
                    '</td><td>'.join(str(_) for _ in row)
                ) for row in data)
                       )
        )
    )


def print_heading(value, text):
    display(HTML('<H{0}>{1}</H{0}>'.format(value, text)))


def dict_to_data_frame(d):
    variables = list(d[0].keys())
    return pd.DataFrame([[i[j] for j in variables] for i in d], columns=variables)


class ClusteringProcessor:
    def __init__(self, data):
        self.data = data
        self.countries = data['Country']
        self.X = data.drop('Country', axis=1)
        self.metrics_info = {}
        self.time_info = {}

    def test_clustering(self):
        for clusters in range(6, 9):
            self.test_k_means(clusters)
            self.test_spectral_clustering(clusters)
            self.test_agglomerative_clustering(clusters)
            for i in range(1, 4):
                threshold = 10 ** -i
                self.test_birch(clusters, threshold)
        self.test_affinity_propagation()
        self.test_mean_shift()
        self.test_dbscan()
        self.plot_metrics_info()

    def display_info(self):
        transformers = [decomposition.DictionaryLearning,
                        decomposition.FactorAnalysis,
                        decomposition.FastICA,
                        decomposition.IncrementalPCA,
                        decomposition.KernelPCA,
                        decomposition.NMF,
                        decomposition.PCA,
                        decomposition.RandomizedPCA,
                        decomposition.SparsePCA,
                        decomposition.TruncatedSVD]

        for tr in transformers:
            plot_transformer = tr(n_components=2).fit(X=self.X)
            plot_data = plot_transformer.transform(self.X)
            for clusters in range(6, 9):
                self.display_k_means(plot_data, clusters, tr)
                self.display_spectral_clustering(plot_data, clusters, tr)
                self.display_agglomerative_clustering(plot_data, clusters, tr)
                for i in range(1, 4):
                    threshold = 10 ** -i
                    self.display_birch(plot_data, clusters, tr, threshold)
            self.display_affinity_propagation(plot_data, tr)
            self.display_mean_shift(plot_data, tr)
            self.display_dbscan(plot_data, tr)

    def plot_clusters(self, estimator, points, title='', can_predict=True,
                      has_cluster_centers=True):
        print_heading(2, title)

        x_min, x_max = points[:, 0].min() - 1, points[:, 0].max() + 1
        y_min, y_max = points[:, 1].min() - 1, points[:, 1].max() + 1
        if can_predict:
            h = .02

            xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

            Z = estimator.predict(np.c_[xx.ravel(), yy.ravel()])

            Z = Z.reshape(xx.shape)
            plt.figure(1)
            plt.clf()

            plt.imshow(Z, interpolation='nearest',
                       extent=(xx.min(), xx.max(), yy.min(), yy.max()),
                       cmap=plt.cm.Paired,
                       aspect='auto', origin='lower')

            plt.plot(points[:, 0], points[:, 1], 'k.', markersize=2)
        else:
            plt.scatter(points[:, 0], points[:, 1], c=estimator.labels_, s=20, edgecolors='w')

        if has_cluster_centers:
            centroids = estimator.cluster_centers_
            plt.scatter(centroids[:, 0], centroids[:, 1],
                        marker='x', s=169, linewidths=3,
                        color='w', zorder=10)

        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.xticks(())
        plt.yticks(())
        plt.show()

    def display_affinity_matrix(self, estimator):
        print_heading(2, 'Affinity matrix')

        fig = plt.figure(figsize=(50, 50))
        ax = fig.add_subplot(111)
        cax = ax.matshow(estimator.affinity_matrix_, interpolation='nearest')
        fig.colorbar(cax)
        plt.xticks(self.countries.index.values, [''] + self.countries, rotation=90, size=10)
        plt.yticks(self.countries.index.values, [''] + self.countries, size=10)
        plt.show()

    def display_labels(self, estimator, title):
        self.display_params(estimator)
        labels = estimator.labels_

        info = pd.DataFrame(self.countries)
        info['cluster'] = labels

        grouped = info.groupby('cluster')['Country'].apply(list)

        info = np.array(list(map(lambda g: [g[0], ', '.join(g[-1]), len(g[-1])], grouped.items())))

        print_heading(2, 'Clusters info')
        print_table('<tr><th>Кластер</th><th>Країни</th><th>Кількість</th></tr>', info)

        print_heading(2, 'Clusters distribution')
        plt.barh(info[:, 0].astype(int), info[:, -1].astype(int), align='center', alpha=0.4)
        plt.yticks(info[:, 0].astype(int))
        plt.show()

        if np.unique(labels).size > 1:
            possible_metrics = ['cityblock', 'euclidean', 'l1', 'l2', 'manhattan',
                                'braycurtis', 'canberra', 'chebyshev', 'hamming',
                                'jaccard', 'mahalanobis', 'minkowski', 'seuclidean',
                                'sqeuclidean']

            print_heading(3, 'Silhouette score:')

            metrics_table_data = []
            for metric in possible_metrics:
                score = metrics.silhouette_score(self.X, labels, metric=metric, sample_size=None)
                self.metrics_info.setdefault(metric, {})
                self.metrics_info[metric][title] = score
                metrics_table_data.append((metric, score))
            print_table('<tr><th>Метрика</th><th>Значення</th></tr>', metrics_table_data)

    def update_fit_time(self, estimator, title):
        start = time.time()
        estimator.fit(X=self.X)
        end = time.time()
        fit_time = end - start
        print_heading(1, '{}, fit time={:0.3f}'.format(title, fit_time))
        self.time_info[title] = fit_time

    def display_k_means(self, x, clusters, transformer):
        km = cluster.KMeans(n_jobs=-2, n_clusters=clusters)
        km.fit(x)
        title = "KMeans, {}, {} clusters".format(transformer.__name__, clusters)
        self.plot_clusters(km, x, title=title)

    def test_k_means(self, clusters):
        title = "KMeans, {} clusters".format(clusters)
        km = cluster.KMeans(n_jobs=-2, n_clusters=clusters)
        self.update_fit_time(km, title)
        self.display_labels(km, title)

    def display_affinity_propagation(self, x, transformer):
        ap = cluster.AffinityPropagation()
        ap.fit(x)
        title = "AffinityPropagation, {}".format(transformer.__name__)
        self.plot_clusters(ap, x, title=title)

    def test_affinity_propagation(self):
        title = "AffinityPropagation"
        ap = cluster.AffinityPropagation()
        self.update_fit_time(ap, title)
        self.display_affinity_matrix(ap)
        self.display_labels(ap, title)

    def display_spectral_clustering(self, x, clusters, transformer):
        sc = cluster.SpectralClustering(n_clusters=clusters)
        sc.fit(x)
        title = "SpectralClustering, {}, {} clusters".format(transformer.__name__, clusters)
        self.plot_clusters(sc, x, title=title, can_predict=False, has_cluster_centers=False)

    def test_spectral_clustering(self, clusters):
        title = "SpectralClustering, {} clusters".format(clusters)
        sc = cluster.SpectralClustering(n_clusters=clusters)
        self.update_fit_time(sc, title)
        self.display_affinity_matrix(sc)
        self.display_labels(sc, title)

    def display_mean_shift(self, x, transformer):
        ms = cluster.MeanShift(n_jobs=-2)
        ms.fit(x)
        title = "MeanShift, {}".format(transformer.__name__)
        self.plot_clusters(ms, x, title=title)

    def test_mean_shift(self):
        title = "MeanShift"
        ms = cluster.MeanShift(n_jobs=-2)
        self.update_fit_time(ms, title)
        self.display_labels(ms, title)

    def display_dbscan(self, x, transformer):
        dbscan = cluster.DBSCAN()
        dbscan.fit(x)
        title = "DBSCAN, {}".format(transformer.__name__)
        self.plot_clusters(dbscan, x, title=title, can_predict=False, has_cluster_centers=False)

    def test_dbscan(self):
        dbscan = cluster.DBSCAN()
        title = "DBSCAN"
        self.update_fit_time(dbscan, title)
        self.display_labels(dbscan, title)

    def display_agglomerative_clustering(self, x, clusters, transformer):
        ac = cluster.AgglomerativeClustering(n_clusters=clusters)
        ac.fit(x)
        title = "AgglomerativeClustering, {}, {} clusters".format(transformer.__name__, clusters)
        self.plot_clusters(ac, x, title=title, can_predict=False, has_cluster_centers=False)

    def test_agglomerative_clustering(self, clusters):
        title = "AgglomerativeClustering, {} clusters".format(clusters)
        ac = cluster.AgglomerativeClustering(n_clusters=clusters)
        self.update_fit_time(ac, title)
        self.display_labels(ac, title)

    def display_birch(self, x, clusters, transformer, threshold):
        brc = cluster.Birch(n_clusters=clusters, threshold=threshold)
        brc.fit(x)
        title = "Birch, {}, {} clusters, threshold={}".format(transformer.__name__, clusters, threshold)
        self.plot_clusters(brc, x, title=title, has_cluster_centers=False)

    def test_birch(self, clusters, threshold):
        title = "Birch, {} clusters, threshold={}".format(clusters, threshold)
        brc = cluster.Birch(n_clusters=clusters, threshold=threshold)
        self.update_fit_time(brc, title)
        self.display_labels(brc, title)

    def plot_metrics_info(self):
        def prepare_info(dictionary):
            return dict_to_data_frame(list(map(lambda i: {'title': i[0], 'value': i[-1]}, dictionary.items())))

        def plot_hist(title, df):
            df = df.sort_values(by='value', ascending=False).reset_index()
            print_heading(3, title)
            print_table('''<tr>
                <th>Метод кластеризації</th>
                <th>Значення</th>
            </tr>''', df[['title', 'value']].values)
            plt.barh(df.index.values, df['value'], align='center', alpha=0.4)
            plt.yticks(df.index.values, df['title'])
            plt.show()

        ti = prepare_info(self.time_info)
        mi = list(map(lambda i: {'name': i[0], 'value': prepare_info(i[-1])}, self.metrics_info.items()))
        plot_hist("Time consumed", ti)
        print_heading(2, "Metrics")
        for item in mi:
            plot_hist(item['name'], item['value'])

    def display_params(self, estimator):
        if hasattr(estimator, 'cluster_centers_'):
            print_heading(3, 'Cluster centers table')
            centers = estimator.cluster_centers_
            mean = centers.mean(axis=0)

            data_to_display = np.append(np.array([np.arange(centers.shape[0])]).T, centers, axis=1)
            data_to_display = pd.DataFrame(data_to_display)
            data_to_display[0] = data_to_display[0].astype(int)
            data_to_display = data_to_display.append([np.append(['Mean'], mean)])
            data_to_display.columns = np.append(['cluster'], self.X.columns)
            data_to_display = pd.concat([data_to_display['cluster'],
                                         data_to_display[self.X.columns].astype(float).applymap(
                                             lambda x: '{:.4f}'.format(x).rstrip('0'))], axis=1)

            print_table('''<tr>
                <th>Cluster</th>
                <th>{}</th>
            </tr>'''.format('</th><th>'.join(self.X.columns)), data_to_display.values)

            print_heading(3, 'Cluster centers plot')
            widths = np.linspace(1, 0, centers.shape[0] + 2, endpoint=False)
            indexes = np.arange(centers.shape[1])
            width = np.diff(widths)[0]
            plt.figure(figsize=(15, 20))
            plt.barh(indexes + (1 - width), mean, width, label='Mean', color='k')
            for i in range(len(centers)):
                plt.barh(indexes + widths[i], centers[i], width,
                         label='Cluster {}'.format(i), color=plt.cm.Pastel2(widths[i]))

            plt.xticks(np.linspace(centers.min(), centers.max(), 20), rotation=90, size=10)
            plt.yticks(np.arange(centers.shape[1]) + 0.5, self.X.columns)
            plt.xlim(centers.min(), centers.max() + (centers.mean() / 2))
            plt.legend(prop={'size': 10}, framealpha=0.5, loc='best')
            plt.show()


if __name__ == '__main__':
    test = ClusteringProcessor(data)
    test.test_clustering()
