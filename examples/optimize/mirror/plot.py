import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
import numpy as np



def trunc(values, decs=0):
    return np.trunc(values*10**decs)/(10**decs)


def main():
    df = pd.read_csv("Factorial_mirror.csv", index_col=0)
    cols = ["light_height","light_width","mirror_offset","inter_0","inter_1","inter_2", "inter_3", "metric"]

    df = df[df["metric"] < 0.110]
    df["metric"] = trunc(df["metric"].to_numpy(), 3)
    fig = px.scatter_3d(df, x=cols[0], y=cols[1], z=cols[2], color=cols[-1], hover_data=df.columns) # range_color=[0.1, 0.15]
    fig.show()


    dimensions = [dict(label=col, values=df[col]) for col in ["light_height","light_width","mirror_offset","inter_0","inter_1", "metric"]]
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
    fig.show()


if __name__ == "__main__":
    main()
