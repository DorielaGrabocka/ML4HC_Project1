import numpy as np
from sklearn.linear_model import LogisticRegression


def get_ensemble_predictions(models, data, n_classes, weights=None):
    """
    Ensembles predictions at the probability predictions level
    weights: if given, should be a list of floats (that sum to 1!) representing the weight of each model in the ensemble
    """
    if weights:
        assert sum(weights) == 1, "Invalid ensembling weights"

    ensemble_preds = np.zeros((len(data), n_classes))
    for idx, model in enumerate(models):
        preds = model.predict_proba(data)
        cur_weight = 1
        if weights:
            cur_weight = weights[idx]

        ensemble_preds += cur_weight * preds

    return ensemble_preds / len(models)


def get_logreg_ensemble_predictions(list_of_models, dataset, dataset_labels, test_set):
    """
     Logistic regression on outputs ensemble method. It requires an array of models and the dataset where it
     should make the predictions, as well as the labels. It returns the logistic regression prediction for each class,
     for each sample as an numpy array for the test set.
     """
    predictions = []
    train_data = np.array(get_list_of_concatenated_predictions(list_of_models, dataset))
    test_data = np.array(get_list_of_concatenated_predictions(list_of_models, test_set))
    log_reg = LogisticRegression(multi_class='multinomial')
    log_reg.fit(train_data, dataset_labels)
    predictions = log_reg.predict_proba(test_data)
    print("Coefficients of Logistic Regression Ensemble: ")
    print(log_reg.coef_)
    return np.array(predictions)


def get_list_of_concatenated_predictions(list_of_models, dataset):
    """Helper method for above function"""
    list_of_predictions = []
    list_of_concatenated_predictions = []
    for model in list_of_models:
        pred = model.predict_proba(dataset)
        list_of_predictions.append(pred)
    # print("Pred-single")
    # print(list_of_predictions)
    list_of_concatenated_predictions = np.hstack(list_of_predictions)
    # print("Pred-total")
    # print(list_of_concatenated_predictions)
    return list_of_concatenated_predictions
