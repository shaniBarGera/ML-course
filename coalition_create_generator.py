from collections import Counter

from numpy.core.multiarray import ndarray
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from data_preproccesor import DF, np, prepare_data
from sklearn.covariance import EmpiricalCovariance
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB as GNB
from coalition_creator_cluster import make_parties_dist
import warnings
warnings.simplefilter("ignore")


# yield list with one's for the items that were selected
def powerset(s):
    x = len(s)
    for i in range(1, (1 << x) - 1):
        yield [s[j] for j in range(x) if (i & (1 << j))]


def is_coalition_majority(data_with_coalition_label: DF):
    coalition_ratio = sum(data_with_coalition_label['Vote']) / len(data_with_coalition_label['Vote'])
    return coalition_ratio > 0.5


def get_data_with_coalition_label(data: DF, coalition)-> DF:
    _data = data.copy()
    _data['Vote'] = data['Vote'].apply(lambda party: 1 if party in coalition else 0)
    return _data


def get_coalition_covariance_norm(data: DF, coalition):
    _data = get_data_with_coalition_label(data, coalition)
    if not is_coalition_majority(_data):
        return 0
    coalition_voters = _data[_data['Vote'] == 1].drop(['Vote'], axis=1)
    covariance: ndarray = EmpiricalCovariance().fit(coalition_voters).covariance_
    return np.linalg.norm(covariance, 'fro')


def get_coalition_score(train: DF, coalition, chosen_model):
    _train = get_data_with_coalition_label(train, coalition)
    if not is_coalition_majority(_train):
        return 0
    x_train = train.drop(['Vote'], axis=1)
    y_train = train['Vote']
    model = chosen_model.fit(x_train, y_train)
    coalition_covariance = model.covariance_[1]
    coalition_mean = model.means_[1]
    oppsition_mean = model.means_[0]
    mean_distance = np.linalg.norm(oppsition_mean - coalition_mean)
    coalition_covariance_norm = np.linalg.norm(coalition_covariance)
    return coalition_covariance_norm + mean_distance*9


def best_coalition(train_data: DF, model):
    parties = train_data['Vote'].unique()
    return max(list(powerset(parties)), key=lambda coalition: get_coalition_score(train_data, coalition, model))


def choose_model(train):
    print('cross validation:')
    generative_models = {'qda': QDA(store_covariance=True), 'lda': LDA(store_covariance=True), 'gnb': GNB()}
    chosen_model = {'Name': 'qda', 'accuracy': 0}
    for model_name, model in generative_models.items():
        acc = cross_validation(train, model)
        print(model_name, acc)
        if acc >= chosen_model['accuracy']:
            chosen_model['Name'] = model_name
            chosen_model['accuracy'] = acc
    print('chosen model:', chosen_model['Name'], chosen_model['accuracy'])
    return generative_models[chosen_model['Name']]


def cross_validation(train: DF, model):
    kf = KFold(n_splits=10)
    accuracy = 0
    x_train = train.drop(['Vote'], axis=1)
    y_train = train['Vote']
    for k, (train_index, test_index) in enumerate(kf.split(train)):
        model.fit(x_train.loc[train_index], y_train.loc[train_index])
        pred = model.predict(x_train.loc[test_index])
        accuracy += accuracy_score(y_train.loc[test_index], pred)
    return accuracy / 10


def best_coalition_generative(train):
    model = choose_model(train)
    coalition = best_coalition(train, model)
    make_parties_dist(train, coalition)
    return coalition
