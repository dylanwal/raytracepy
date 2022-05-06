
import pandas as pd
import plotly.express as px


def main():
    df = pd.read_csv("mirror/combinations.csv")
    df = df.loc[df["number_lights"]==49]
    fig = px.scatter_3d(df, x="width", y="height", z="mean", color="mirror_offset")
    fig.show()
    fig = px.scatter_3d(df, x="width", y="height", z="std", color="mirror_offset")
    fig.show()


if __name__ == "__main__":
    main()
