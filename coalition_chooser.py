from coalition_create_generator import best_coalition_generative
from coalition_creator_cluster import best_coalition_clustering
from sklearn.metrics import davies_bouldin_score
from data_preproccesor import DF, np, prepare_data


def get_coalition_score(data, coalition_parties):
    _data = data.copy()
    _data['Vote'] = data['Vote'].apply(lambda party: 1 if party in coalition_parties else 0)
    return davies_bouldin_score(_data, _data['Vote'])


if __name__ == '__main__':
    train, validation, test = prepare_data()
    cluster_coalition = best_coalition_clustering(train)
    generative_coalition = best_coalition_generative(train)
    generative_score = get_coalition_score(train, generative_coalition)
    clustering_score = get_coalition_score(train, cluster_coalition)
    print('generative_score:', generative_score)
    print('clustering_score:', clustering_score)
