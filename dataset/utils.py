import numpy as np
import pandas as pd
from itertools import combinations


def feature_name_combiner(col, value) -> str:
    def replace(s):
        return s.replace("<", "lt_").replace(">", "gt_").replace("=", "eq_").replace("[", "lb_").replace("]", "ub_")

    col = replace(str(col))
    value = replace(str(value))
    return f'{col}="{value}"'


def feature_name_restorer(feature_name) -> str:
    return (
        feature_name.replace("lt_", "<").replace("gt_", ">").replace("eq_", "=").replace("lb_", "[").replace("ub_", "]")
    )