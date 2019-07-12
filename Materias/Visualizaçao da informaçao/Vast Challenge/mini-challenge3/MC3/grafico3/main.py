import pandas as pd
import numpy as np
from bokeh.plotting import figure, show, output_file
from bokeh.io import output_notebook, reset_output, curdoc
from bokeh.models import LabelSet, Label, ColumnDataSource, FactorRange, HoverTool
from bokeh.models.annotations import Title
import bokeh.plotting as bp
from bokeh.layouts import row, column,layout, gridplot
from bokeh.models.widgets import Slider, TextInput, RangeSlider, Select
from bokeh.models import ColumnDataSource
from bokeh.palettes import Spectral6
from bokeh.plotting import figure
from bokeh.transform import factor_cmap

data = pd.read_csv("grafico3/data/data_retweets.csv", index_col=0,parse_dates=True, infer_datetime_format=True)
data = data.sort_index()
data["x"] = np.arange(data.shape[0])
data["time"] = data.index

p = figure(plot_height=450, width=1300, title="re-tweets", background_fill_color="#F7F7F7",
           tools="hover,box_select,pan,wheel_zoom,box_zoom,reset,save")
source = ColumnDataSource(dict(data))
p.vbar(x='x', top='num_re', width=1, source=source,line_color="white",fill_color="green")
p.y_range.start = 0
p.x_range.range_padding = 0.1
p.xgrid.grid_line_color = None
p.hover.mode = "vline"
p.hover.tooltips = [("data", "@time{%F %T}"),
                    ("chave", "@word"),
                    ("tweet", "@message")]

p.hover.formatters = {'time': 'datetime'}
p.yaxis.axis_label= "número de re-tweets"

curdoc().add_root(row([p]))
