import glob
import time

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px


def _load_files(file_name: str) -> list[pd.DataFrame]:
    files = glob.glob(file_name)
    out = []
    for file in files:
        out.append(pd.read_csv(file, index_col=0))
    return out


def list_of_df_to_numpy(list_df: list[pd.DataFrame], col: str) -> np.ndarray:
    min_length = min([df.shape[0] for df in list_df])
    data = np.empty((len(list_df), min_length))
    for i, df in enumerate(list_df):
        data[i, :] = df[col].to_numpy()[:min_length]

    return data


def _get_best(df: pd.DataFrame, upper_lower: str = "upper", col: str = "metric"):
    best_col = np.empty_like(df.index, dtype="float64")
    if upper_lower == "upper":
        best_value = max(df[col])
    else:
        best_value = min(df[col])
    for i, val in enumerate(df[col]):
        if upper_lower == "upper":
            if val < best_value:
                best_value = val
        else:
            if val > best_value:
                best_value = val

        best_col[i] = best_value

    df["best"] = best_col


def _average_by_expt(list_df: list[pd.DataFrame], col: str = "metric"):
    return np.mean(list_of_df_to_numpy(list_df, col), axis=0)


def _get_bounds(list_df: list[pd.DataFrame], col: str = "metric") -> (np.ndarray, np.ndarray):
    data = list_of_df_to_numpy(list_df, col)

    lb = np.min(data, axis=0)
    ub = np.max(data, axis=0)

    return lb, ub


class VizBatch:

    def __init__(self, file_name: str):
        self.data = _load_files(file_name)
        self.df = pd.concat(self.data, ignore_index=True)

        self._var_names = None
        self._iter_names = None
        self._get_names()

    @property
    def var_names(self) -> list[str]:
        return self._var_names

    @property
    def inter_names(self) -> list[str]:
        return self._iter_names

    def _get_names(self):
        columns = self.df.columns.tolist()
        if "iteration" in columns:
            columns.remove("iteration")
        columns.remove("metric")

        vars = []
        inters = []
        for name in columns:
            if name.startswith("inter_"):
                inters.append(name)
            else:
                vars.append(name)

        self._var_names = vars
        self._inter_names = inters

    def plot_by_expt(self) -> go.Figure:
        fig = go.Figure()
        for df in self.data:
            self._plot_by_expt(fig, df)

        x = self.data[0].index.to_numpy()
        fig.add_trace(
            go.Scatter(x=x, y=_average_by_expt(self.data, "metric"), name="best", legendgroup="best", showlegend=False,
                       mode="markers+lines", line=dict(width=3, color="black"),
                       fill='tonexty'
                       )
        )

        return fig

    def plot_by_expt_best(self, best_value: float = None) -> go.Figure:
        fig = go.Figure()

        for df in self.data:
            _get_best(df)

        min_length = min([df.shape[0] for df in self.data])
        x = np.linspace(1, min_length, min_length)
        lb, ub = _get_bounds(self.data, "best")

        if best_value is not None:
            fig.add_trace(go.Scatter(x=[0, x[-1]], y=[best_value, best_value], name= "global min",
                                     mode="lines", line=dict(width=2, color='red', dash='dash')))

        fig.add_trace(
            go.Scatter(x=x, y=lb, name="lower_bound", legendgroup="best", showlegend=False,
                       mode="lines", line=dict(width=1, color="gray"),
                       )
        )
        fig.add_trace(
            go.Scatter(x=x, y=ub, name="upper_bound", legendgroup="best", showlegend=False,
                       mode="lines", line=dict(width=1, color="gray"),
                       fill='tonexty'
                       )
        )
        fig.add_trace(
            go.Scatter(x=x, y=_average_by_expt(self.data, "best"), name="average", legendgroup="best", showlegend=True,
                       mode="markers+lines", line=dict(width=3, color="black"),
                       )
        )

        # formatting
        fig.update_layout(autosize=False, width=800, height=600, font=dict(family="Arial", size=18, color="black"),
                          plot_bgcolor="white", showlegend=True, legend=dict(x=.75, y=.95))
        fig.update_xaxes(title="<b>function evaluation </b>", tickprefix="<b>", ticksuffix="</b>", showline=True,
                         linewidth=5, mirror=True, linecolor='black', ticks="outside", tickwidth=4, showgrid=False,
                         gridwidth=1, gridcolor="lightgray", range=[0, 25])
        fig.update_yaxes(title="<b>std/mean</b>", tickprefix="<b>", ticksuffix="</b>", showline=True,
                         linewidth=5, mirror=True, linecolor='black', ticks="outside", tickwidth=4, showgrid=False,
                         gridwidth=1, gridcolor="lightgray", range=[0.15, 0.3])

        return fig

    @staticmethod
    def _plot_by_expt(fig: go.Figure, df: pd.DataFrame, col: str = "metric"):
        fig.add_trace(
            go.Scatter(x=df.index, y=df[col], name=col,
                       mode="markers+lines", line=dict(width=1)
                       ))

    def plot_4d_vis(self, indept_var: list[str] = None, metric_vis: str = "color", cut_off: float=None, **kwargs) -> go.Figure:
        if len(self.df) < 4:
            raise ValueError("3 independent variable are required to visualized in 4D.")

        if indept_var is None:
            cols = self.var_names[:3]
        else:
            if len(indept_var) != 3:
                raise ValueError("Three independent variable names must be provided")
            cols = indept_var

        if metric_vis == "color":
            cols.append("metric")
        elif metric_vis == "z":
            cols.insert(2, "metric")
        else:
            raise ValueError("Invalid metric_vis value.")

        if cut_off is not None:
            df = self.df[self.df["metric"] < cut_off]
        else:
            df = self.df

        fig = px.scatter_3d(df, x=cols[0], y=cols[1], z=cols[2], color=cols[3], hover_data=self.df.columns, **kwargs)
        return fig

    def matrix_plot(self):
        dimensions = [dict(label=col, values=self.df[col]) for col in self.df.columns]
        fig = go.Figure(data=go.Splom(
                    dimensions=dimensions,
                    showupperhalf=False, # remove plots on diagonal
                    diagonal=dict(visible=False),
                    hoverinfo="all"
                    # text=self.df['class'],
                    # marker=dict(color=index_vals,
                    #             showscale=False, # colors encode categorical variables
                    #             line_color='white', line_width=0.5)
                    ))
        return fig

    def replace_columns(self, replacements: list[list[str]]):
        columns = self.df.columns
        for replace in replacements:
            columns = [replace[1] if x == replace[0] else x for x in columns]

        self.df.columns = columns
        for data in self.data:
            data.columns = columns


def main():
    # file_name = "no_mirror/nelder_mead/MethodNelderMead_no_mirror_*.csv"
    # file_name = "no_mirror/sobol/MethodSobol_no_mirror_*.csv"
    # file_name = "no_mirror/powell/MethodPowell_no_mirror_*.csv"
    file_name = "no_mirror/dragonfly/MethodBODragon_no_mirror_*.csv"
    file_name = "no_mirror/hypercube/MethodLatinHypercube_no_mirror_*.csv"

    vizbatch = VizBatch(file_name)
    vizbatch.replace_columns([["inter_0", "mean"], ["inter_1", "std"], ["inter_2", "p10"], ["inter_3", "p90"]])
    vizbatch.df = vizbatch.df.drop(columns=["p10", "p90"])

    fig = vizbatch.plot_by_expt_best(best_value=0.166)
    fig.show()

    # time.sleep(1)
    # fig1 = vizbatch.plot_4d_vis() # cut_off=10.2, range_color=(0.165, 0.167)
    # fig1.show()

    # time.sleep(1)
    # vizbatch.df = vizbatch.df.drop(columns=["iteration"])
    # fig2 = vizbatch.matrix_plot()
    # fig2.show()


if __name__ == "__main__":
    main()
