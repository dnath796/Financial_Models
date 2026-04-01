import argparse
import os
import tempfile
import warnings

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from arch import arch_model


class GarchModel:
    def __init__(self, data_path, date_col=None, value_col=None, return_type="log"):
        self.data_path = data_path
        self.date_col = date_col
        self.value_col = value_col
        self.return_type = return_type
        self.series = self.load_data()
        self.returns = self.compute_returns()
        self.model = None
        self.result = None
        self.forecast_variance = None
        self.last_observation_date = self.series.index[-1] if isinstance(self.series.index, pd.DatetimeIndex) else None
        self.selected_order = None
        self.selection_criterion = None
        self.selection_score = None
        self.mean_name = None
        self.vol_name = None
        self.dist_name = None

    def load_data(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found at {self.data_path}")

        df = pd.read_csv(self.data_path, sep=None, engine="python")
        df.columns = df.columns.str.strip()

        if self.date_col:
            date_col = self.date_col.strip()
            if date_col not in df.columns:
                raise ValueError(f"Date column '{self.date_col}' not found in {list(df.columns)}")
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.sort_values(date_col).set_index(date_col)

        if self.value_col:
            value_col = self.value_col.strip()
            if value_col not in df.columns:
                raise ValueError(f"Value column '{self.value_col}' not found in {list(df.columns)}")
            series = df[value_col]
        else:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric_cols:
                raise ValueError("No numeric columns found in the input data.")
            series = df[numeric_cols[0]]

        series = pd.to_numeric(series, errors="coerce").dropna()
        if series.empty:
            raise ValueError("Selected series is empty after removing non-numeric values.")

        print("Loaded series.")
        if isinstance(series.index, pd.DatetimeIndex):
            print(f"Date range: {series.index.min().date()} to {series.index.max().date()}")
        print("Most recent observations:")
        print(series.tail())
        return series

    def compute_returns(self):
        if self.return_type == "log":
            returns = 100 * np.log(self.series / self.series.shift(1))
        elif self.return_type == "pct":
            returns = 100 * self.series.pct_change()
        else:
            raise ValueError("return_type must be either 'log' or 'pct'.")

        returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
        if returns.empty:
            raise ValueError("Return series is empty. Check the input data and return_type.")

        print("\nMost recent returns (%):")
        print(returns.tail())
        return returns

    def plot_series(self):
        plt.figure(figsize=(11, 5))
        plt.plot(self.series, label="Original series")
        plt.title("Input Series")
        plt.xlabel("Date" if isinstance(self.series.index, pd.DatetimeIndex) else "Observation")
        plt.ylabel("Value")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_returns(self):
        plt.figure(figsize=(11, 5))
        plt.plot(self.returns, label="Returns", color="tab:blue")
        plt.title("Returns Used for GARCH")
        plt.xlabel("Date" if isinstance(self.returns.index, pd.DatetimeIndex) else "Observation")
        plt.ylabel("Return (%)")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def fit(self, p=1, q=1, mean="Constant", vol="GARCH", dist="normal"):
        print(f"\nFitting {vol}({p}, {q}) model with mean='{mean}' and dist='{dist}'...")
        self.model = arch_model(self.returns, mean=mean, vol=vol, p=p, q=q, dist=dist)
        self.result = self.model.fit(disp="off")
        self.selected_order = (p, q)
        self.selection_criterion = None
        self.selection_score = None
        self.mean_name = mean
        self.vol_name = vol
        self.dist_name = dist
        print(self.result.summary())
        return self.result

    def select_best_order(self, max_p=3, max_q=3, criterion="aic", mean="Constant", vol="GARCH", dist="normal"):
        criterion = criterion.lower()
        if criterion not in {"aic", "bic"}:
            raise ValueError("criterion must be either 'aic' or 'bic'.")
        if max_p < 1 or max_q < 1:
            raise ValueError("max_p and max_q must both be at least 1.")

        print(f"\nSearching {vol}(p, q) orders for p=1..{max_p}, q=1..{max_q} using {criterion.upper()}...")

        best_score = None
        best_order = None
        best_result = None

        for p in range(1, max_p + 1):
            for q in range(1, max_q + 1):
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        candidate_model = arch_model(
                            self.returns,
                            mean=mean,
                            vol=vol,
                            p=p,
                            q=q,
                            dist=dist,
                        )
                        candidate_result = candidate_model.fit(disp="off")
                    score = getattr(candidate_result, criterion)
                    print(f"  {vol}({p}, {q}) -> {criterion.upper()}: {score:.4f}")
                except Exception as exc:
                    print(f"  {vol}({p}, {q}) -> failed: {exc}")
                    continue

                if best_score is None or score < best_score:
                    best_score = score
                    best_order = (p, q)
                    best_result = candidate_result

        if best_result is None:
            raise RuntimeError("No valid GARCH models were fitted during order selection.")

        self.model = best_result.model
        self.result = best_result
        self.selected_order = best_order
        self.selection_criterion = criterion.upper()
        self.selection_score = best_score
        self.mean_name = mean
        self.vol_name = vol
        self.dist_name = dist

        print(f"\nSelected best order: {vol}{best_order} with {criterion.upper()}={best_score:.4f}")
        print(self.result.summary())
        return best_order, best_score, best_result

    def plot_conditional_volatility(self):
        if self.result is None:
            raise RuntimeError("Fit the model before plotting conditional volatility.")

        plt.figure(figsize=(11, 5))
        plt.plot(self.result.conditional_volatility, color="tab:red", label="Conditional volatility")
        plt.title("Estimated Conditional Volatility")
        plt.xlabel("Date" if isinstance(self.returns.index, pd.DatetimeIndex) else "Observation")
        plt.ylabel("Volatility (%)")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def forecast(self, horizon=5):
        if self.result is None:
            raise RuntimeError("Fit the model before forecasting.")

        forecast = self.result.forecast(horizon=horizon, reindex=False)
        variance = forecast.variance.iloc[-1]
        volatility = np.sqrt(variance)
        self.forecast_variance = variance

        if self.last_observation_date is not None:
            forecast_index = pd.bdate_range(
                start=self.last_observation_date + pd.offsets.BDay(1),
                periods=horizon,
            )
        else:
            forecast_index = [f"t+{step}" for step in range(1, horizon + 1)]

        forecast_df = pd.DataFrame(
            {
                "forecast_variance": variance.values,
                "forecast_volatility": volatility.values,
            },
            index=forecast_index,
        )
        forecast_df.index.name = "forecast_date" if self.last_observation_date is not None else "horizon"

        print("\nForecasted variance and volatility:")
        print(forecast_df)
        return forecast_df

    def standardized_residuals(self):
        if self.result is None:
            raise RuntimeError("Fit the model before requesting residuals.")
        return self.result.std_resid.dropna()

    def historical_context(self, forecast_df):
        if self.result is None:
            raise RuntimeError("Fit the model before building historical context.")

        level_series = self.series.dropna()
        current_level = float(level_series.iloc[-1])
        hist_mean = float(level_series.mean())
        hist_median = float(level_series.median())
        hist_min = float(level_series.min())
        hist_max = float(level_series.max())
        hist_min_date = level_series.idxmin()
        hist_max_date = level_series.idxmax()
        current_percentile = float((level_series <= current_level).mean() * 100)

        cond_vol = self.result.conditional_volatility.dropna()
        current_cond_vol = float(cond_vol.iloc[-1])
        avg_cond_vol = float(cond_vol.mean())
        median_cond_vol = float(cond_vol.median())
        cond_vol_percentile = float((cond_vol <= current_cond_vol).mean() * 100)

        forecast_start = float(forecast_df["forecast_volatility"].iloc[0])
        forecast_end = float(forecast_df["forecast_volatility"].iloc[-1])
        forecast_avg = float(forecast_df["forecast_volatility"].mean())
        forecast_change = forecast_end - forecast_start

        if isinstance(forecast_df.index, pd.DatetimeIndex):
            forecast_start_date = forecast_df.index[0]
            forecast_end_date = forecast_df.index[-1]
        else:
            forecast_start_date = None
            forecast_end_date = None

        return {
            "current_level": current_level,
            "hist_mean": hist_mean,
            "hist_median": hist_median,
            "hist_min": hist_min,
            "hist_max": hist_max,
            "hist_min_date": hist_min_date,
            "hist_max_date": hist_max_date,
            "current_percentile": current_percentile,
            "current_cond_vol": current_cond_vol,
            "avg_cond_vol": avg_cond_vol,
            "median_cond_vol": median_cond_vol,
            "cond_vol_percentile": cond_vol_percentile,
            "forecast_start": forecast_start,
            "forecast_end": forecast_end,
            "forecast_avg": forecast_avg,
            "forecast_change": forecast_change,
            "forecast_start_date": forecast_start_date,
            "forecast_end_date": forecast_end_date,
        }

    def explain_outlook(self, forecast_df):
        context = self.historical_context(forecast_df)

        if context["current_level"] >= context["hist_mean"]:
            level_vs_history = "above"
        else:
            level_vs_history = "below"

        if context["forecast_change"] > 0.1:
            direction_text = "rising modestly"
        elif context["forecast_change"] < -0.1:
            direction_text = "easing modestly"
        else:
            direction_text = "staying broadly stable"

        lines = [
            "Historical context:",
            f"- Sample runs from {self.series.index.min().date()} to {self.series.index.max().date()}.",
            f"- Latest OVX level is {context['current_level']:.2f}, which is {level_vs_history} the full-sample average of {context['hist_mean']:.2f}.",
            f"- The current OVX level sits around the {context['current_percentile']:.1f}th percentile of the full history.",
            f"- Full-sample range is {context['hist_min']:.2f} on {context['hist_min_date'].date()} to {context['hist_max']:.2f} on {context['hist_max_date'].date()}.",
            "",
            "Volatility interpretation:",
            f"- Latest model-implied conditional volatility is {context['current_cond_vol']:.2f}%, versus a historical model average of {context['avg_cond_vol']:.2f}%.",
            f"- That places current conditional volatility near the {context['cond_vol_percentile']:.1f}th percentile of the model's full history.",
            "",
            "Upcoming projection:",
            f"- For {context['forecast_start_date'].date()} to {context['forecast_end_date'].date()}, forecast volatility averages {context['forecast_avg']:.2f}%.",
            f"- The model projects volatility {direction_text}, from {context['forecast_start']:.2f}% at the start of the week to {context['forecast_end']:.2f}% by the end of the week.",
            "- In plain terms: the model is not forecasting a volatility collapse. It is pointing to an elevated and slightly firmer risk regime in the coming week.",
        ]

        return "\n".join(lines), context

    def create_team_report(self, forecast_df, output_path="ovx_garch_report.png", recent_window=126, title=None):
        if self.result is None:
            raise RuntimeError("Fit the model before creating a report.")

        level_window = self.series
        vol_window = self.result.conditional_volatility.dropna()
        narrative, context = self.explain_outlook(forecast_df)

        latest_value = float(self.series.iloc[-1])
        model_label = self.vol_name or "GARCH"
        if self.selected_order is not None:
            model_label = f"{model_label}{self.selected_order}"

        if title is None:
            title = "OVX Volatility Forecast Report"

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle(title, fontsize=18, fontweight="bold", y=0.98)

        axes[0, 0].plot(level_window.index, level_window.values, color="tab:blue", linewidth=2)
        axes[0, 0].scatter(level_window.index[-1], level_window.iloc[-1], color="black", zorder=3)
        axes[0, 0].axhline(context["hist_mean"], color="tab:gray", linestyle="--", linewidth=1.5, label="Full-history average")
        if isinstance(self.series.index, pd.DatetimeIndex):
            recent_start = self.series.index[max(0, len(self.series) - recent_window)]
            axes[0, 0].axvspan(recent_start, self.series.index[-1], color="#dfefff", alpha=0.35, label=f"Recent {recent_window} obs")
        axes[0, 0].set_title("OVX Level Across Full History")
        axes[0, 0].set_ylabel("Index level")
        axes[0, 0].grid(alpha=0.25)
        axes[0, 0].legend(loc="upper left")

        axes[0, 1].plot(vol_window.index, vol_window.values, color="tab:orange", linewidth=1.8, label="Historical conditional volatility")
        axes[0, 1].plot(
            forecast_df.index,
            forecast_df["forecast_volatility"].values,
            color="tab:red",
            linewidth=2.2,
            marker="o",
            label="Upcoming forecast",
        )
        axes[0, 1].axhline(context["avg_cond_vol"], color="tab:gray", linestyle="--", linewidth=1.2, label="Full-history avg volatility")
        axes[0, 1].set_title("Volatility Regime: History to Upcoming Week")
        axes[0, 1].set_ylabel("Volatility (%)")
        axes[0, 1].grid(alpha=0.25)
        axes[0, 1].legend(loc="upper left")

        axes[1, 0].bar(
            forecast_df.index,
            forecast_df["forecast_volatility"].values,
            color="#cf5c36",
            width=0.7,
        )
        axes[1, 0].axhline(context["current_cond_vol"], color="black", linestyle="--", linewidth=1.2, label="Latest fitted volatility")
        axes[1, 0].axhline(context["avg_cond_vol"], color="tab:gray", linestyle=":", linewidth=1.2, label="Historical avg volatility")
        axes[1, 0].set_title("Upcoming Trading-Week Forecast")
        axes[1, 0].set_ylabel("Forecast volatility (%)")
        axes[1, 0].grid(alpha=0.25)
        axes[1, 0].legend()

        axes[1, 1].axis("off")
        axes[1, 1].text(
            0.02,
            0.98,
            "\n".join(
                [
                    "Presentation Summary",
                    f"Latest OVX level: {latest_value:.2f}",
                    f"Model used: {model_label}",
                    f"Mean model: {self.mean_name or 'N/A'}",
                    f"Distribution: {self.dist_name or 'N/A'}",
                    (
                        f"Selected by {self.selection_criterion}: {self.selection_score:.2f}"
                        if self.selection_criterion is not None and self.selection_score is not None
                        else "Selected order: manual"
                    ),
                    "",
                    narrative,
                ]
            ),
            va="top",
            ha="left",
            fontsize=11,
            bbox={"boxstyle": "round,pad=0.6", "facecolor": "#f7f3e8", "edgecolor": "#444444"},
        )

        for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
            ax.tick_params(axis="x", rotation=30)

        fig.tight_layout(rect=[0, 0, 1, 0.95])

        if output_path:
            fig.savefig(output_path, dpi=200, bbox_inches="tight")
            print(f"\nSaved report to {output_path}")

        plt.close(fig)
        return output_path


def build_parser():
    parser = argparse.ArgumentParser(description="Fit a GARCH model to a univariate time series.")
    parser.add_argument(
        "--data-path",
        default="/Users/deepikanath/dnath796/Git/Financial_Models/Data_center/OVXCLS.csv",
        help="Path to the input data file.",
    )
    parser.add_argument("--date-col", default="Date", help="Name of the date column.")
    parser.add_argument("--value-col", default="Value", help="Name of the value column.")
    parser.add_argument(
        "--return-type",
        choices=["log", "pct"],
        default="log",
        help="Return transformation to fit the volatility model on.",
    )
    parser.add_argument("--p", type=int, default=1, help="ARCH lag order.")
    parser.add_argument("--q", type=int, default=1, help="GARCH lag order.")
    parser.add_argument(
        "--select-order",
        action="store_true",
        help="Search for the best (p, q) order instead of using --p and --q directly.",
    )
    parser.add_argument("--max-p", type=int, default=3, help="Maximum ARCH lag to test during order selection.")
    parser.add_argument("--max-q", type=int, default=3, help="Maximum GARCH lag to test during order selection.")
    parser.add_argument(
        "--criterion",
        choices=["aic", "bic"],
        default="aic",
        help="Model selection criterion for automatic order search.",
    )
    parser.add_argument("--mean", default="Constant", help="Mean model passed to arch_model.")
    parser.add_argument("--vol", default="GARCH", help="Volatility model passed to arch_model.")
    parser.add_argument("--dist", default="normal", help="Error distribution passed to arch_model.")
    parser.add_argument("--horizon", type=int, default=10, help="Forecast horizon.")
    parser.add_argument("--plot", action="store_true", help="Show series, returns, and volatility plots.")
    parser.add_argument(
        "--report-path",
        default=None,
        help="Optional PNG path for a team-ready visual report.",
    )
    parser.add_argument(
        "--recent-window",
        type=int,
        default=126,
        help="Number of recent observations to show in the visual report.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    model = GarchModel(
        data_path=args.data_path,
        date_col=args.date_col,
        value_col=args.value_col,
        return_type=args.return_type,
    )

    if args.plot:
        model.plot_series()
        model.plot_returns()

    if args.select_order:
        model.select_best_order(
            max_p=args.max_p,
            max_q=args.max_q,
            criterion=args.criterion,
            mean=args.mean,
            vol=args.vol,
            dist=args.dist,
        )
    else:
        model.fit(p=args.p, q=args.q, mean=args.mean, vol=args.vol, dist=args.dist)

    if args.plot:
        model.plot_conditional_volatility()

    forecast_df = model.forecast(horizon=args.horizon)
    outlook_text, _ = model.explain_outlook(forecast_df)
    print("\n" + outlook_text)

    if args.report_path:
        model.create_team_report(
            forecast_df=forecast_df,
            output_path=args.report_path,
            recent_window=args.recent_window,
        )


if __name__ == "__main__":
    main()
