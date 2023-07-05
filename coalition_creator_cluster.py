from data_preproccesor import DF, np, prepare_data
from collections import Counter
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt


def get_clustered_model(data: DF, algo, n_clusters=2):
    x = data.drop(['Vote'], axis=1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=1, algorithm=algo).fit(x)
    return kmeans


def get_clusters(data, algo):
    kmeans = get_clustered_model(data, algo)
    data.loc[:, 'cluster'] = kmeans.labels_
    cluster1 = data[data['cluster'] == 0].drop(['cluster'], axis=1)
    cluster2 = data[data['cluster'] == 1].drop(['cluster'], axis=1)

    clusters = [Counter(cluster1['Vote']), Counter(cluster2['Vote'])]
    cluster1_dist = max(clusters, key=lambda cluster: sum(cluster.values()))
    cluster2_dist = min(clusters, key=lambda cluster: sum(cluster.values()))
    return cluster1_dist, cluster2_dist


def get_coalition(cluster1, cluster2, parties)-> np.array:
    group1_dist = {}
    group2_dist = {}

    for party in parties:
        # print(party, ":", cluster1_dist[party], cluster2_dist[party])
        party_prec = cluster1[party]/(cluster1[party] + cluster2[party])
        if party_prec > 0.95:
            group1_dist[party] = cluster1[party] + cluster2[party]
        else:
            group2_dist[party] = cluster1[party] + cluster2[party]

    return list(group1_dist.keys())


def validate_coalition(test_data: DF, coalition, num_parties):
    test_data['Vote'] = test_data['Vote'].apply(lambda party: 1 if party in coalition else 0)
    coalition_voters = test_data[test_data['Vote'] == 1]
    coalition_density: float = get_clustered_model(coalition_voters, 'auto', 1).inertia_
    coalition_size = len(coalition)
    size_ratio = coalition_size / num_parties
    #print("coalition size ratio:", size_ratio)
    #print("coalition_density:", coalition_density, "1 / coalition_density :", 1 / coalition_density)
    return 1 / coalition_density


def k_cross_validation(train_data: DF, algo):
    kf = KFold(n_splits=10)
    qual_vec = []
    print("algorithm:", algo)
    coalition = []
    parties = train_data['Vote'].unique()
    num_parties = len(parties)
    for k, (train_index, test_index) in enumerate(kf.split(train_data)):
        c1, c2 = get_clusters(train_data.loc[train_index], algo)
        coalition = get_coalition(c1, c2, parties)
        #print("coalition:", coalition)
        coalition_quality = validate_coalition(train_data.loc[test_index], coalition, num_parties)
        #print("total quality:", coalition_quality, "\n")
        qual_vec.append(coalition_quality)
    print("score:", sum(qual_vec)/len(qual_vec), '\n')
    return sum(qual_vec)/len(qual_vec)


def choose_algo(train_data: DF):
    algos = ['auto', 'full', 'elkan']
    print("cross validation:")
    algo_index = np.argmax([k_cross_validation(train_data, algo) for algo in algos])
    return algos[algo_index]


def make_parties_dist(train, party_list):
    parties_dist = Counter(train['Vote'])
    party_list_dist = {key: parties_dist[key] for key in party_list}
    total_votes = sum(parties_dist.values())
    list_votes = sum(party_list_dist.values())
    print(party_list_dist, list_votes, list_votes/total_votes)


def plot_clusters(c1, c2, parties):
    dict = {}
    for party in parties:
        dict[party] = [c1[party], c2[party]]
    DF(dict).to_csv('results/clusters.csv')


def best_coalition_clustering(train):
    algo = choose_algo(train)
    parties = train['Vote'].unique()
    cluster1, cluster2 = get_clusters(train, algo)
    plot_clusters(cluster1, cluster2, parties)
    coalition = get_coalition(cluster1, cluster2, parties)
    make_parties_dist(train, coalition)
    return coalition
