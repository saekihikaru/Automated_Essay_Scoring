import lightgbm as lgb
import xgboost as xgb
import catboost as cbt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_X_y
from sklearn.metrics import log_loss
from sklearn.svm import SVC
from sklearn.utils import check_X_y
from sklearn.metrics import accuracy_score

from .base_model import BaseClassifier

from .utils import f1_micro_lgb

import numpy as np
from sklearn.metrics import cohen_kappa_score

def kappa_metric(y_true, y_pred):
    y_pred = np.expand_dims(y_pred, axis=1)
    y_pred_labels = np.argmax(y_pred, axis=1)
    kappa = cohen_kappa_score(y_true, y_pred_labels)
    return kappa

class KappaMetric:
    def get_final_error(self, error, weight):
        return error

    def is_max_optimal(self):
        return True

    def evaluate(self, approxes, target, weight):
        """
        Calculate the Quadratic Weighted Kappa.
        
        :param approxes: List of approximate predictions for each class
        :param target: List of true target labels
        :param weight: List of weights for each sample
        :return: Quadratic Weighted Kappa score
        """

        print(target)
        print(approxes)
        exit()
        y_pred = np.argmax(approxes, axis=0)
        y_true = target.tolist() if isinstance(target, np.ndarray) else list(target)
        y_pred = y_pred.tolist() if isinstance(y_pred, np.ndarray) else list(y_pred)

        min_rating = min(min(y_true), min(y_pred))
        max_rating = max(max(y_true), max(y_pred))
        N = int(max_rating - min_rating + 1)

        O = np.zeros((N, N))
        for a, p in zip(y_true, y_pred):
            O[int(a - min_rating)][int(p - min_rating)] += 1

        w = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                w[i][j] = ((i - j) ** 2) / ((N - 1) ** 2)
        
        act_hist = np.zeros(N)
        for item in y_true:
            act_hist[int(item - min_rating)] += 1

        pred_hist = np.zeros(N)
        for item in y_pred:
            pred_hist[int(item - min_rating)] += 1

        E = np.outer(act_hist, pred_hist)
        E = E / E.sum()

        O = O / O.sum()

        num = np.sum(w * O)
        den = np.sum(w * E)

        return 1 - (num / den)







class XGBoostClassifier(BaseClassifier):
    def __init__(self, input_dim, output_dim, model_config, verbose, seed=None) -> None:
        super().__init__(input_dim, output_dim, model_config, verbose)
        self.model = xgb.XGBClassifier(
            objective="multi:softprob",
            eval_metric="mlogloss",
            early_stopping_rounds=50,
            **self.model_config,
            random_state=seed
        )

    def fit(self, X, y, eval_set):
        self._column_names = X.columns
        X, y = check_X_y(X, y)
        eval_set = [(eval_set[0].values, eval_set[1])]  # eval_setを適切な形式に変更

        y -= y.min()
        eval_set = [(eval_X, eval_y - eval_y.min()) for eval_X, eval_y in eval_set]
        self.model.fit(X, y, eval_set=eval_set, verbose=self.verbose > 0)
        # print("XGBoostモデルの使用されたパラメータ:", self.model.get_params())

    def feature_importance(self):
        return self.model.feature_importances_

class LightGBMClassifier(BaseClassifier):
    def __init__(self, input_dim, output_dim, model_config, verbose, seed=None) -> None:
        super().__init__(input_dim, output_dim, model_config, verbose)

        self.model = lgb.LGBMClassifier(
            objective="multiclass",  # 2値分類用のobjectiveに変更
            verbose=self.verbose,
            random_state=seed,
            **self.model_config,
        )

    def fit(self, X, y, eval_set):
        self._column_names = X.columns
        X, y = check_X_y(X, y)
        eval_set = [(eval_set[0].values, eval_set[1])]  # eval_setを適切な形式に変更

        self.model.fit(
            X,
            y,
            eval_set=eval_set,
            eval_metric='multi_logloss',
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=self.verbose > 0)],
        )
        # print("LightGBMモデルの使用されたパラメータ:", self.model.get_params())
        
    def feature_importance(self):
        return self.model.feature_importances_
    
class CBTClassifier(BaseClassifier):
    def __init__(self, input_dim, output_dim, model_config, verbose, seed=42) -> None:
        super().__init__(input_dim, output_dim, model_config, verbose)

        self.model = cbt.CatBoostClassifier(
            loss_function="MultiClass",  # 2値分類の場合は "Logloss"
            verbose=self.verbose,
            random_seed=seed,
            **self.model_config,
            # eval_metric="MultiClass"
        )

    def fit(self, X, y, eval_set):
        self._column_names = X.columns

        # Set custom metric
        # self.model.set_params(custom_metric=[KappaMetric()])

        self.model.fit(
            X,
            y,
            eval_set=[eval_set],
        )

    def feature_importance(self):
        return self.model.feature_importances_

class SVMClassifier(BaseClassifier):
    def __init__(self, input_dim, output_dim, model_config, verbose, seed=None) -> None:
        super().__init__(input_dim, output_dim, model_config, verbose)
        
        self.model = SVC(
            probability=True,  # 確率推定を有効にする
            
            random_state=seed,
            **self.model_config
        )

    def fit(self, X, y, eval_set):
        self._column_names = X.columns
        X, y = check_X_y(X, y)
        X_val, y_val = eval_set[0].values, eval_set[1]

        self.model.fit(X, y)
        
        # # ここではSVMには早期終了機能がないため、評価セットに対する予測を行う
        # y_pred_val = self.model.predict(X_val)
        # val_score = accuracy_score(y_val, y_pred_val)
        # if self.verbose:
        #     print(f'Validation Accuracy: {val_score}')

class AdaBoostCustomClassifier(BaseClassifier):
    def __init__(self, input_dim, output_dim, model_config, verbose, seed=None) -> None:
        super().__init__(input_dim, output_dim, model_config, verbose)
        
        base_estimator=DecisionTreeClassifier(max_depth=3)  # 決定木を基本分類器として使用

        # Adaboost分類器の定義
        self.model = AdaBoostClassifier(
            estimator=base_estimator,  # 決定木を基本分類器として使用
            # n_estimators=50,  # Adaboostの基本モデルの数
            # num_clfs=5,
            random_state=seed,
            **model_config,  # その他のモデル設定を適用
        )

    def fit(self, X, y, eval_set):
        self._column_names = X.columns
        X, y = check_X_y(X, y)
        eval_set = [(eval_set[0].values, eval_set[1])]  # eval_setを適切な形式に変更

        # Adaboost分類器の学習
        self.model.fit(X, y)

    def feature_importance(self):
        return self.model.feature_importances_