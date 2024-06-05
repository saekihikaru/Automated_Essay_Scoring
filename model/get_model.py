from experiment.utils import set_seed

from .model import LightGBMClassifier, XGBoostClassifier, CBTClassifier, SVMClassifier, AdaBoostCustomClassifier


def get_classifier(name, *, input_dim, output_dim, model_config, seed=42, early_stopping=50, verbose=0):
    set_seed(seed=seed)
    if name == "xgboost":
        return XGBoostClassifier(input_dim, output_dim, model_config, verbose, seed)
    elif name == "lightgbm":
        return LightGBMClassifier(input_dim, output_dim, model_config, verbose, seed)
    elif name == "catboost":
        return CBTClassifier(input_dim, output_dim, model_config, verbose, seed)
    elif name == "svm":
        return SVMClassifier(input_dim, output_dim, model_config, verbose, seed)
    elif name == "adaboost":
        return AdaBoostCustomClassifier(input_dim, output_dim, model_config, verbose, seed)
    else:
        raise KeyError(f"{name} is not defined.")