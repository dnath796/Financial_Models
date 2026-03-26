import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
from statsmodels.base.model import GenericLikelihoodModel
from statsmodels.discrete.discrete_model import Logit, Probit
from statsmodels.tools.sm_exceptions import ConvergenceWarning, PerfectSeparationWarning


DATA_PATH = "/Users/deepikanath/dnath796/Git/Financial_Models/P/Financial_Modelling/airbnb.csv"
LEFT_CENSOR = 1.0
EPS = 1e-12


class TobitModel(GenericLikelihoodModel):
    def __init__(self, endog, exog, left_censor=LEFT_CENSOR, **kwargs):
        self.left_censor = left_censor
        super().__init__(endog, exog, **kwargs)

    def loglike(self, params):
        beta = params[:-1]
        sigma = max(np.abs(params[-1]), EPS)

        xb = self.exog @ beta
        y = self.endog

        censored = y <= self.left_censor
        uncensored = ~censored
        ll = np.zeros(len(y))

        if uncensored.any():
            resid = (y[uncensored] - xb[uncensored]) / sigma
            ll[uncensored] = norm.logpdf(resid) - np.log(sigma)

        if censored.any():
            cdf_vals = norm.cdf((self.left_censor - xb[censored]) / sigma)
            ll[censored] = np.log(np.clip(cdf_vals, EPS, 1.0))

        return ll.sum()


def print_header(title):
    print(f"\n=== {title} ===")


def build_design_matrix(frame, columns):
    x = sm.add_constant(frame[columns], has_constant="add").astype(float)
    assert not x.isna().any().any(), "NaNs in design matrix"
    assert np.isfinite(x.values).all(), "Infinite values in design matrix"
    return x


def fit_binary_model(model_cls, y, x, label):
    counts = y.value_counts(dropna=False).sort_index()
    print(f"{label} counts: {counts.to_dict()}")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", category=PerfectSeparationWarning)
        warnings.simplefilter("always", category=ConvergenceWarning)
        result = model_cls(y, x).fit(disp=0)

    model_warnings = [
        warning for warning in caught
        if issubclass(warning.category, (PerfectSeparationWarning, ConvergenceWarning))
    ]
    if model_warnings:
        print(f"Warnings for {label}:")
        for warning in model_warnings:
            print(f"- {warning.category.__name__}: {warning.message}")

    return result


def fit_tobit(y, x, left_censor=LEFT_CENSOR):
    censored_mask = y <= left_censor
    print(f"Tobit censoring counts: {pd.Series(censored_mask).value_counts().to_dict()}")

    uncensored_mask = ~censored_mask
    x_values = x.to_numpy()
    y_values = y.to_numpy()

    ols_start = np.linalg.lstsq(x_values[uncensored_mask], y_values[uncensored_mask], rcond=None)[0]
    init_sigma = np.std(y_values[uncensored_mask] - x_values[uncensored_mask] @ ols_start)
    init_params = np.append(ols_start, max(init_sigma, 1.0))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", category=ConvergenceWarning)
        warnings.simplefilter("always", category=UserWarning)
        model = TobitModel(y, x, left_censor=left_censor)
        result = model.fit(start_params=init_params, disp=0)

    model_warnings = [
        warning for warning in caught
        if issubclass(warning.category, (ConvergenceWarning, UserWarning))
    ]
    if model_warnings:
        print("Warnings for Tobit:")
        for warning in model_warnings:
            print(f"- {warning.category.__name__}: {warning.message}")

    return result


def fit_heckman_two_step(y_selection, z_selection, y_outcome, x_outcome):
    probit_sel = Probit(y_selection, z_selection).fit(disp=0)
    xb = probit_sel.predict(z_selection, which="linear")

    cdf_vals = np.clip(norm.cdf(xb), EPS, 1.0)
    imr = pd.Series(norm.pdf(xb) / cdf_vals, index=z_selection.index)

    observed_mask = y_selection == 1
    x_second_stage = x_outcome.loc[observed_mask].copy()
    x_second_stage["lambda"] = imr.loc[observed_mask]
    y_second_stage = y_outcome.loc[observed_mask]

    outcome_model = sm.OLS(y_second_stage, x_second_stage).fit()
    return probit_sel, outcome_model


data = pd.read_csv(DATA_PATH)
data["selection"] = data["minimum_stay"].notna().astype(int)

base_columns = [
    "price",
    "rating",
    "reviews",
    "room_type",
    "accommodates",
    "bedrooms",
]

selection_data = data.dropna(subset=base_columns + ["selection"]).copy()
selection_data = pd.get_dummies(selection_data, columns=["room_type"], drop_first=True)

room_type_cols = [col for col in selection_data.columns if col.startswith("room_type_")]
feature_cols = ["price", "rating", "reviews", "accommodates", "bedrooms"] + room_type_cols

z_sel = build_design_matrix(selection_data, feature_cols)
y_sel = selection_data["selection"].astype(int)

observed_data = selection_data.loc[selection_data["selection"] == 1].copy()
x_out = build_design_matrix(observed_data, feature_cols)
y_out = observed_data["minimum_stay"].astype(float)

print_header("Model Setup")
print("Outcome variable: minimum_stay")
print("Selection variable: 1 if minimum_stay is observed, 0 if missing")
print(f"Rows loaded: {len(data)}")
print(f"Rows in selection sample: {len(selection_data)}")
print(f"Rows in observed outcome sample: {len(observed_data)}")
print(f"Selection counts: {y_sel.value_counts().to_dict()}")
print(f"Minimum stay value counts: {y_out.value_counts().sort_index().to_dict()}")
print(f"Tobit left-censor point: {LEFT_CENSOR}")

print_header("Logit Results")
logit_result = fit_binary_model(Logit, y_sel, z_sel, "Selection variable")
print(logit_result.summary())

print_header("Probit Results")
probit_result = fit_binary_model(Probit, y_sel, z_sel, "Selection variable")
print(probit_result.summary())

print_header("Tobit Results")
tobit_result = fit_tobit(y_out, x_out, left_censor=LEFT_CENSOR)
print(tobit_result.summary())

print_header("OLS Outcome Model")
ols_result = sm.OLS(y_out, x_out).fit()
print(ols_result.summary())

print_header("Heckman 2-Step Results")
heckman_probit, heckman_ols = fit_heckman_two_step(y_sel, z_sel, selection_data["minimum_stay"], z_sel)
print("Step 1: Probit selection equation")
print(heckman_probit.summary())
print("\nStep 2: OLS outcome equation with IMR")
print(heckman_ols.summary())
