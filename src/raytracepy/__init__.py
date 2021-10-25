number_type = "float64"


def default_plot_layout(fig):
    fig.update_layout(autosize=False, width=900, height=790, showlegend=False,
                      font=dict(family="Arial", size=18, color="black"), plot_bgcolor="white")
    fig.update_xaxes(title="<b>X<b>", tickprefix="<b>", ticksuffix="</b>", showline=True, linewidth=0, mirror=True,
                     linecolor='black', showgrid=False, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(title="<b>Y<b>", tickprefix="<b>", ticksuffix="</b>", showline=True, mirror=True, linewidth=0,
                     linecolor='black', showgrid=False, gridwidth=1, gridcolor='lightgray')


from .light_layouts import CirclePattern, GridPattern

__all__ = ["CirclePattern", "GridPattern"]