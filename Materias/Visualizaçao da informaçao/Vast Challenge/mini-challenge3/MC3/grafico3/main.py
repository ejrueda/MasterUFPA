import pandas as pd
import numpy as np
from bokeh.plotting import figure, show, output_file
from bokeh.io import output_notebook, reset_output, curdoc
from bokeh.models import LabelSet, Label, ColumnDataSource, FactorRange, HoverTool, ColorBar
from bokeh.models.annotations import Title
import bokeh.plotting as bp
from bokeh.layouts import row, column,layout, gridplot
from bokeh.models.widgets import Slider, TextInput, RangeSlider, Select
from bokeh.models import ColumnDataSource
from bokeh.palettes import Spectral11
from bokeh.plotting import figure
from bokeh.transform import factor_cmap, linear_cmap

mapper = linear_cmap(field_name="num_re", palette=Spectral11, low=0, high=76)
color_bar = ColorBar(
        color_mapper = mapper['transform'], 
        width        = 8,
        location     = (0,0))

data = pd.read_csv("grafico3/data/data_retweets.csv", index_col=0,parse_dates=True, infer_datetime_format=True)
data = data.sort_index()
data["time"] = data.index
data = data.loc[data.num_re>5]
data["x"] = np.arange(data.shape[0])
for idx, val in data.iterrows():
    data.loc[idx,"x_ticks"] = pd.to_datetime(str(val.time)).strftime('%m/%d@%H:%M')
d1 = {val.x: val.x_ticks for i, val in data.iterrows()}
p = figure(plot_height=450, width=1800, title="re-tweets", background_fill_color="#F7F7F7",
           tools="hover,box_select,pan,wheel_zoom,box_zoom,reset,save")
source = ColumnDataSource(dict(data))
p.vbar(x='x', top='num_re', width=1, source=source,line_color="black",fill_color=mapper)
p.y_range.start = 0
p.x_range.range_padding = 0.1
p.xgrid.grid_line_color = None
p.hover.mode = "vline"
p.hover.tooltips = [("data", "@time{%F %T}"),
                    ("chave", "@word"),
                    ("conta", "@account"),
      		    ("location", "@location"),
		    ("re-tweets", "@num_re"),
                    ("tweet", "@message")]

p.hover.formatters = {'time': 'datetime'}
p.yaxis.axis_label= "número de re-tweets"
p.hover.point_policy ="follow_mouse"
p.xaxis.major_label_overrides = d1
p.add_layout(color_bar, "right")

data_water = pd.read_csv("grafico3/data/data_water.csv",parse_dates=True, infer_datetime_format=True)
data_shake = pd.read_csv("grafico3/data/data_shake.csv",parse_dates=True, infer_datetime_format=True)
data_rain = pd.read_csv("grafico3/data/data_rain.csv",parse_dates=True, infer_datetime_format=True)
data_power = pd.read_csv("grafico3/data/data_power.csv",parse_dates=True, infer_datetime_format=True)
data_hsptal = pd.read_csv("grafico3/data/data_hsptal.csv",parse_dates=True, infer_datetime_format=True)
data_build = pd.read_csv("grafico3/data/data_build.csv",parse_dates=True, infer_datetime_format=True)
data_bridge = pd.read_csv("grafico3/data/data_bridge.csv",parse_dates=True, infer_datetime_format=True)
data_tsum = pd.read_csv("grafico3/data/data_tsum.csv",parse_dates=True, infer_datetime_format=True)
data_circus = pd.read_csv("grafico3/data/data_circus.csv",parse_dates=True, infer_datetime_format=True)
data_music = pd.read_csv("grafico3/data/data_music.csv",parse_dates=True, infer_datetime_format=True)

for idx, val in data_shake.iterrows():
    data_shake.loc[idx,"x_ticks"] = pd.to_datetime(str(val.time)).strftime('%m/%d@%H:%M')
for idx, val in data_power.iterrows():
    data_power.loc[idx,"x_ticks"] = pd.to_datetime(str(val.time)).strftime('%m/%d@%H:%M')
for idx, val in data_bridge.iterrows():
    data_bridge.loc[idx,"x_ticks"] = pd.to_datetime(str(val.time)).strftime('%m/%d@%H:%M')
for idx, val in data_hsptal.iterrows():
    data_hsptal.loc[idx,"x_ticks"] = pd.to_datetime(str(val.time)).strftime('%m/%d@%H:%M')
for idx, val in data_build.iterrows():
    data_build.loc[idx,"x_ticks"] = pd.to_datetime(str(val.time)).strftime('%m/%d@%H:%M')
for idx, val in data_water.iterrows():
    data_water.loc[idx,"x_ticks"] = pd.to_datetime(str(val.time)).strftime('%m/%d@%H:%M')
for idx, val in data_rain.iterrows():
    data_rain.loc[idx,"x_ticks"] = pd.to_datetime(str(val.time)).strftime('%m/%d@%H:%M')
for idx, val in data_tsum.iterrows():
    data_tsum.loc[idx,"x_ticks"] = pd.to_datetime(str(val.time)).strftime('%m/%d@%H:%M')
for idx, val in data_circus.iterrows():
    data_circus.loc[idx,"x_ticks"] = pd.to_datetime(str(val.time)).strftime('%m/%d@%H:%M')
for idx, val in data_music.iterrows():
    data_music.loc[idx,"x_ticks"] = pd.to_datetime(str(val.time)).strftime('%m/%d@%H:%M')

d2 = {val.x: val.x_ticks for i, val in data_shake.iterrows()}
d3 = {val.x: val.x_ticks for i, val in data_power.iterrows()}
d4 = {val.x: val.x_ticks for i, val in data_bridge.iterrows()}
d5 = {val.x: val.x_ticks for i, val in data_hsptal.iterrows()}
d6 = {val.x: val.x_ticks for i, val in data_build.iterrows()}
d7 = {val.x: val.x_ticks for i, val in data_water.iterrows()}
d8 = {val.x: val.x_ticks for i, val in data_rain.iterrows()}
d9 = {val.x: val.x_ticks for i, val in data_tsum.iterrows()}
d10 = {val.x: val.x_ticks for i, val in data_circus.iterrows()}
d11 = {val.x: val.x_ticks for i, val in data_music.iterrows()}

p2 = figure(plot_height=250, width=900, title="shake", background_fill_color="#F7F7F7",
           tools="hover,box_select,pan,wheel_zoom,box_zoom,reset,save")
source2 = ColumnDataSource(dict(data_shake))
p2.vbar(x='x', top='num_re', width=1, source=source2,fill_color=mapper, line_color='black')
p2.y_range.start = 0
p2.xgrid.grid_line_color = None
p2.hover.mode = "vline"
p2.hover.tooltips = [("data", "@time{%F %T}"),
                    ("chave", "@word"),
                    ("conta", "@account"),
		    ("re-tweets", "@num_re"),
      		    ("location", "@location"),
                    ("tweet", "@message")]
p2.hover.formatters = {'time': 'datetime'}
p2.xaxis.axis_label="time"
p2.yaxis.axis_label="quntidade de retweets"
p2.xaxis.major_label_overrides = d2
p2.x_range.range_padding = 0.01
p2.hover.point_policy ="follow_mouse"
#p2.xaxis.major_label_orientation = 5

p3 = figure(plot_height=250, width=900, title="power", background_fill_color="#F7F7F7",
           tools="hover,box_select,pan,wheel_zoom,box_zoom,reset,save")
source3 = ColumnDataSource(dict(data_power))
p3.vbar(x='x', top='num_re', width=1, source=source3,fill_color=mapper, line_color='black')
p3.y_range.start = 0
p3.xaxis.major_label_overrides = d3
p3.x_range.range_padding = 0.01
p3.xgrid.grid_line_color = None
p3.hover.mode = "vline"
p3.hover.tooltips = [("data", "@time{%F %T}"),
                    ("chave", "@word"),
                    ("conta", "@account"),
	   	    ("re-tweets", "@num_re"),
        	    ("location", "@location"),
                    ("tweet", "@message")]
p3.hover.formatters = {'time': 'datetime'}
p3.xaxis.axis_label="time"
p3.yaxis.axis_label="quntidade de retweets"
p3.hover.point_policy ="follow_mouse"

p4 = figure(plot_height=250, width=900, title="bridge", background_fill_color="#F7F7F7",
           tools="hover,box_select,pan,wheel_zoom,box_zoom,reset,save")
source4 = ColumnDataSource(dict(data_bridge))
p4.vbar(x='x', top='num_re', width=1, source=source4,fill_color=mapper, line_color='black')
p4.y_range.start = 0
p4.xaxis.major_label_overrides = d4
p4.x_range.range_padding = 0.01
p4.xgrid.grid_line_color = None
p4.hover.mode = "vline"
p4.hover.tooltips = [("data", "@time{%F %T}"),
                    ("chave", "@word"),
                    ("conta", "@account"),
		    ("re-tweets", "@num_re"),
        	    ("location", "@location"),
                    ("tweet", "@message")]
p4.hover.formatters = {'time': 'datetime'}
p4.xaxis.axis_label="time"
p4.yaxis.axis_label="quntidade de retweets"
p4.hover.point_policy ="follow_mouse"

p5 = figure(plot_height=250, width=900, title="hospital", background_fill_color="#F7F7F7",
           tools="hover,box_select,pan,wheel_zoom,box_zoom,reset,save")
source5 = ColumnDataSource(dict(data_hsptal))
p5.vbar(x='x', top='num_re', width=1, source=source5,fill_color=mapper, line_color='black')
p5.y_range.start = 0
p5.xaxis.major_label_overrides = d5
p5.x_range.range_padding = 0.01
p5.xgrid.grid_line_color = None
p5.hover.mode = "vline"
p5.hover.tooltips = [("data", "@time{%F %T}"),
                    ("chave", "@word"),
                    ("conta", "@account"),
		    ("re-tweets", "@num_re"),
        	    ("location", "@location"),
                    ("tweet", "@message")]
p5.hover.formatters = {'time': 'datetime'}
p5.xaxis.axis_label="time"
p5.yaxis.axis_label="quntidade de retweets"
p5.hover.point_policy ="follow_mouse"

p6 = figure(plot_height=250, width=900, title="build", background_fill_color="#F7F7F7",
           tools="hover,box_select,pan,wheel_zoom,box_zoom,reset,save", x_range=(-3,3))
source6 = ColumnDataSource(dict(data_build))
p6.vbar(x='x', top='num_re', width=0.3, source=source6,fill_color=mapper, line_color='black')
p6.y_range.start = 0
p6.xaxis.major_label_overrides = d6
#p6.x_range.range_padding = 0.01
p6.xgrid.grid_line_color = None
p6.hover.mode = "vline"
p6.hover.tooltips = [("data", "@time{%F %T}"),
                    ("chave", "@word"),
                    ("conta", "@account"),
	   	    ("re-tweets", "@num_re"),
        	    ("location", "@location"),
                    ("tweet", "@message")]
p6.hover.formatters = {'time': 'datetime'}
p6.xaxis.axis_label="time"
p6.yaxis.axis_label="quntidade de retweets"
p6.hover.point_policy ="follow_mouse"

p7 = figure(plot_height=250, width=900, title="water", background_fill_color="#F7F7F7",
           tools="hover,box_select,pan,wheel_zoom,box_zoom,reset,save")
source7 = ColumnDataSource(dict(data_water))
p7.vbar(x='x', top='num_re', width=1, source=source7,fill_color=mapper, line_color='black')
p7.y_range.start = 0
p7.xaxis.major_label_overrides = d7
p7.x_range.range_padding = 0.01
p7.xgrid.grid_line_color = None
p7.hover.mode = "vline"
p7.hover.tooltips = [("data", "@time{%F %T}"),
                    ("chave", "@word"),
                    ("conta", "@account"),
		    ("re-tweets", "@num_re"),
        	    ("location", "@location"),
                    ("tweet", "@message")]
p7.hover.formatters = {'time': 'datetime'}
p7.xaxis.axis_label="time"
p7.yaxis.axis_label="quntidade de retweets"
p7.hover.point_policy ="follow_mouse"

p8 = figure(plot_height=250, width=900, title="rain", background_fill_color="#F7F7F7",
           tools="hover,box_select,pan,wheel_zoom,box_zoom,reset,save")
source8 = ColumnDataSource(dict(data_rain))
p8.vbar(x='x', top='num_re', width=1, source=source8,fill_color=mapper, line_color='black')
p8.y_range.start = 0
p8.xaxis.major_label_overrides = d8
p8.x_range.range_padding = 0.01
p8.xgrid.grid_line_color = None
p8.hover.mode = "vline"
p8.hover.tooltips = [("data", "@time{%F %T}"),
                    ("chave", "@word"),
                    ("conta", "@account"),
		    ("re-tweets", "@num_re"),
        	    ("location", "@location"),
                    ("tweet", "@message")]
p8.hover.formatters = {'time': 'datetime'}
p8.xaxis.axis_label="time"
p8.yaxis.axis_label="quntidade de retweets"
p8.hover.point_policy ="follow_mouse"

p9 = figure(plot_height=250, width=900, title="tsunami", background_fill_color="#F7F7F7",
           tools="hover,box_select,pan,wheel_zoom,box_zoom,reset,save")
source9 = ColumnDataSource(dict(data_tsum))
p9.vbar(x='x', top='num_re', width=1, source=source9,fill_color=mapper, line_color='black')
p9.y_range.start = 0
p9.xaxis.major_label_overrides = d9
p9.x_range.range_padding = 0.01
p9.xgrid.grid_line_color = None
p9.hover.mode = "vline"
p9.hover.tooltips = [("data", "@time{%F %T}"),
                    ("chave", "@word"),
                    ("conta", "@account"),
		    ("re-tweets", "@num_re"),
        	    ("location", "@location"),
                    ("tweet", "@message")]
p9.hover.formatters = {'time': 'datetime'}
p9.xaxis.axis_label="time"
p9.yaxis.axis_label="quntidade de retweets"
p9.hover.point_policy ="follow_mouse"

p10 = figure(plot_height=250, width=900, title="circus", background_fill_color="#F7F7F7",
           tools="hover,box_select,pan,wheel_zoom,box_zoom,reset,save")
source10 = ColumnDataSource(dict(data_circus))
p10.vbar(x='x', top='num_re', width=1, source=source10,fill_color=mapper, line_color='black')
p10.y_range.start = 0
p10.xaxis.major_label_overrides = d10
p10.x_range.range_padding = 0.01
p10.xgrid.grid_line_color = None
p10.hover.mode = "vline"
p10.hover.tooltips = [("data", "@time{%F %T}"),
                    ("chave", "@word"),
                    ("conta", "@account"),
		    ("re-tweets", "@num_re"),
        	    ("location", "@location"),
                    ("tweet", "@message")]
p10.hover.formatters = {'time': 'datetime'}
p10.xaxis.axis_label="time"
p10.yaxis.axis_label="quntidade de retweets"
p10.hover.point_policy ="follow_mouse"

p11 = figure(plot_height=250, width=900, title="music", background_fill_color="#F7F7F7",
           tools="hover,box_select,pan,wheel_zoom,box_zoom,reset,save")
source11 = ColumnDataSource(dict(data_music))
p11.vbar(x='x', top='num_re', width=1, source=source11,fill_color=mapper, line_color='black')
p11.y_range.start = 0
p11.xaxis.major_label_overrides = d11
p11.x_range.range_padding = 0.01
p11.xgrid.grid_line_color = None
p11.hover.mode = "vline"
p11.hover.tooltips = [("data", "@time{%F %T}"),
                    ("chave", "@word"),
                    ("conta", "@account"),
		    ("re-tweets", "@num_re"),
        	    ("location", "@location"),
                    ("tweet", "@message")]
p11.hover.formatters = {'time': 'datetime'}
p11.xaxis.axis_label="time"
p11.yaxis.axis_label="quntidade de retweets"
p11.hover.point_policy ="follow_mouse"

row_1 = row([p])
row_2 = row([p2,p3])
row_3 = row([p4,p5])
row_4 = row([p6,p7])
row_5 = row([p8,p9])
row_6 = row([p11,p10])

curdoc().add_root(row(gridplot([[row_1],[row_2],[row_3],[row_4],[row_5],[row_6]]), width=400))
curdoc().title="Re-tweets"
