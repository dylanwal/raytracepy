
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go


def main():
    df = pd.read_csv("nsga2_results.csv")
    df2 = pd.read_csv("nsga2_results2.csv")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["mean"], y=df["std"], mode="markers"))
    fig.add_trace(go.Scatter(x=df2["mean"], y=df2["std"], mode="markers"))
    fig.show()
    print(df)


if __name__ == "__main__":
    main()
