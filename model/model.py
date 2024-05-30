import lightgbm as lgb
import xgboost as xgb
from sklearn.utils.validation import check_X_y

from .base_model import BaseClassifier


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