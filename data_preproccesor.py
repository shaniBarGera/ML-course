import pandas as pd
import numpy as np
from pandas import DataFrame as DF
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import RFECV
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle

train_properties = {}
prepared_data = {}
label_title = "Vote"
features_types = pd.read_csv("data/features_dtype.csv")


def is_numeric(feature):
    return (features_types[feature][0] == 'uint' or
            features_types[feature][0] == 'ufloat' or
            features_types[feature][0] == 'float')


def write_data_to_csv_file(prepared_data: dict, time, result_dir="results/"):
    for data_type, data in prepared_data.items():
        data.to_csv(result_dir + time + data_type + ".csv", index=False)


def convert_features_to_nominal(data):
    categorical_features = data.select_dtypes(['object']).columns
    categorical_features = categorical_features.drop('Vote')
    data[categorical_features] = data[categorical_features].apply(lambda feat: feat.astype("category").cat.codes)


def fill_train_properties(train_data: DF):
    features = train_data.columns
    for feature in features:
        if feature == label_title:
            continue
        train_properties[feature] = dict()
        train_properties[feature]["median"] = train_data[feature].median()
        train_properties[feature]["std"] = train_data[feature].std()
        train_properties[feature]["mean"] = train_data[feature].mean()


def imputation(data: DF):
    for feature in data.columns:
        if is_numeric(feature):
            fill_value = train_properties[feature]["median"]
        else:
            fill_value = data[feature].value_counts().idxmax()
        data[feature].fillna(fill_value, inplace=True)


def detect_outliers(data):
    for feature in data.columns:
        if is_numeric(feature):
            feature_zscores = \
                abs((data[feature] - train_properties[feature]["mean"]) / train_properties[feature]["std"])
            data[feature] = np.where(feature_zscores > 3, np.nan, data[feature])
            data[feature] = np.where(data[feature] < 0, np.nan, data[feature])

def drop_bad_samples(data, outlier_threshold=15):
    samples_to_drop = []
    for index, sample in data.iterrows():
        if sample.isna().sum() > outlier_threshold:
            samples_to_drop.append(index)
    data.drop(samples_to_drop)


def normalize(data: dict):
    uniform_features = ['Yearly_IncomeK', 'Overall_happiness_score', 'Number_of_differnt_parties_voted_for',
                        "Most_Important_Issue"]
    normal_features = [ 'Avg_size_per_room', 'Avg_Satisfaction_with_previous_vote',
                        'Political_interest_Total_Score', 'Weighted_education_rank', 'Avg_monthly_income_all_years']

    standard_scaler = StandardScaler()
    minimax_scaler = MinMaxScaler()

    minimax_scaler.fit(data["train"][uniform_features].astype(np.float))
    standard_scaler.fit(data["train"][normal_features])

    for data_type, curr_data in data.items():
        curr_data[uniform_features] = minimax_scaler.transform(curr_data[uniform_features])
        curr_data[normal_features] = standard_scaler.transform(curr_data[normal_features])


def get_feature_prefix(feature):
    return feature.split("@")[0]


def write_selected_features(selected_features, features):
    is_selected = pd.read_csv("data/SelectedFeaturesTemplate.csv")
    features_selected = []
    for feature in features:
        if feature == 'Vote':
            continue
        # for case of one hot. the word before @ is the feature
        feature_prefix = get_feature_prefix(feature)
        if feature_prefix in features_selected:
            continue
        if feature in selected_features:
            is_selected.loc[0, feature_prefix] = 1
            features_selected.append(feature_prefix)
        else:
            is_selected.loc[0, feature_prefix] = 0
    is_selected.to_csv("results/SelectedFeatures.csv", index=False)


def print_selected_features():
    is_selected = pd.read_csv("results/SelectedFeatures.csv")
    selected_features = []
    for feature in is_selected.columns:
        if feature == "Feature name":
            continue
        if is_selected.loc[0, feature] == 1:
            selected_features.append(feature)
    print("number of selected features:" + str(len(selected_features)))
    print("selected features:")
    print(selected_features)


def convert_category_to_one_hot(data: DF):
    features = data.columns
    for feature in features:
        if feature == 'Vote':
            continue
        if features_types[feature][0] == 'nominal':
            data = pd.concat([data, pd.get_dummies(data[feature], prefix=feature+"@")], axis=1, join='inner')
            data.drop([feature], axis=1, inplace=True)
    return data


def test_data(train, test):
    x_train = train.drop(['Vote'], axis=1)
    y_train = train['Vote']
    x_test = test.drop(['Vote'], axis=1)
    y_test = test['Vote']

    clf = SVC()
    clf = clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    print('accuracy:', metrics.accuracy_score(y_test, y_pred))
    print('f1 score:', metrics.f1_score(y_test, y_pred, average='micro'))
    print()



def prepare_data():
    # Get Data
    raw_data = pd.read_csv("data/ElectionsData.csv")
    raw_data = shuffle(raw_data, random_state=1)
    original_features = raw_data.columns

    # Split Data
    train_percent = 0.75
    validation_percent = 0.15
    test_percent = 0.1
    train, validation, test = np.array_split(raw_data, [int(len(raw_data) * train_percent),
                                                        int(len(raw_data) * (train_percent + validation_percent))])
    orig_data = {"train": train, "validation": validation, "test": test}
    write_data_to_csv_file(orig_data, "old")
    new_data = {"train": train.copy(), "validation": validation.copy(), "test": test.copy()}

    # Select right features
    selected_features = ['Vote', 'Yearly_IncomeK', 'Number_of_differnt_parties_voted_for',
                         'Political_interest_Total_Score', 'Avg_Satisfaction_with_previous_vote',
                         'Avg_monthly_income_all_years', 'Most_Important_Issue', 'Overall_happiness_score',
                         'Avg_size_per_room', 'Weighted_education_rank']
    new_data["train"], new_data["test"], new_data["validation"] = new_data["train"][selected_features], \
                                                                  new_data["test"][selected_features], \
                                                                  new_data["validation"][selected_features]

    # Convert to Numeric
    for data_type, curr_data in new_data.items():
        convert_features_to_nominal(curr_data)

    fill_train_properties(new_data["train"])

    print("imputation and outliers")
    # Outlier Detection
    for data_type, curr_data in new_data.items():
        detect_outliers(curr_data)

    # Imputation
    fill_train_properties(new_data["train"])
    for data_type, curr_data in new_data.items():
        imputation(curr_data)

    print("normalize & nominal conversion")
    # Normalization
    normalize(new_data)

    # Convert nominal to vector
    for data_type, curr_data in new_data.items():
        new_data[data_type] = convert_category_to_one_hot(curr_data)

    # Write results
    write_data_to_csv_file(new_data, "new")
    write_selected_features(new_data["train"].columns, new_data["train"].columns)

    train = pd.read_csv("results/newtrain.csv")
    validation = pd.read_csv("results/newvalidation.csv")
    test = pd.read_csv("results/newtest.csv")
    return train, validation, test
