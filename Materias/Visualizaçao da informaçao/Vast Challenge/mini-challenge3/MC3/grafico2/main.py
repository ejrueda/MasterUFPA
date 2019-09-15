import pandas as pd
import numpy as np
from bokeh.plotting import figure, show, output_file
from bokeh.io import output_notebook, reset_output, curdoc
from bokeh.embed import file_html
from bokeh.models import LabelSet, Label, ColumnDataSource, FactorRange, HoverTool
from bokeh.models.annotations import Title
from bokeh.transform import factor_cmap
import bokeh.plotting as bp
from bokeh.layouts import row, column,layout, gridplot
from bokeh.models.widgets import Slider, TextInput, RangeSlider, Select
from bokeh.transform import cumsum, jitter
from bokeh.palettes import Category10, Category20

#os conjuntos de dados são carrregados
data_pro = pd.read_csv("./grafico2/data/data_geral_normalizada.csv", index_col=0, parse_dates=True,
                       infer_datetime_format=True)
data_cluster = pd.read_csv("./grafico2/data/data_words.csv", index_col=0, infer_datetime_format=True,
                           parse_dates=True)
#bairros
bairros = ["Palace Hills", "Northwest", "Old Town", "Safe Town", "Southwest", "Downtown",
         "Wilson Forest", "Scenic Vista", "Broadview", "Chapparal", "Terrapin Springs",
         "Pepper Mill", "Cheddarford", "Easton", "Weston", "Southton", "Oak Willow",
         "East Parton", "West Parton"]
bairro_init = "Palace Hills"
#gráfico, palavras chaves nos bairros
p1 = figure(plot_width=1150, plot_height=500, y_range=bairros, x_axis_type='datetime',
           title="palavras chave por bairro no tempo",
          tools="hover,pan,wheel_zoom,box_zoom,reset,save",
          background_fill_color="#F7F7F7")
source1 = ColumnDataSource(data_pro)
p1.circle(x='time', y=jitter('location', width=0.7, range=p1.y_range),  source=source1, alpha=0.3)
p1.hover.tooltips = [("palavras", "@message")]
p1.x_range.range_padding = 0.01
p1.ygrid.grid_line_color = None

#gráfico dois, cluster das palavras chave
data_aux = data_cluster.loc[data_cluster.location==bairro_init]
data_aux["x"] = data_aux.index.values
source2 = ColumnDataSource(dict(data_aux))
pvras = ["power","shake","hospital","disaster","sewer","rescue","build","others"]
p2 = figure(plot_width=1150, plot_height=400, y_range=pvras, x_axis_type='datetime',
           title="Palavras chave por bairro",
          tools="hover,pan,wheel_zoom,box_zoom,reset,save",
	          background_fill_color="#F7F7F7", x_range=p1.x_range)
p2.circle(x='x', y=jitter('tipo', width=0.6, range=p2.y_range),  source=source2, alpha=0.5,)
p2.hover.tooltips = [("datetime", "@x{%F %T}"),
                    ("tipo", "@tipo"),
                    ("palavra", "@word")]
p2.hover.formatters = {'x': 'datetime'}
p2.x_range.range_padding = .01
p2.ygrid.grid_line_color = None
p2.xaxis.axis_label = "dias do terremoto"
p2.yaxis.axis_label = "palavras chave no terremoto"

#Barras de interação
select_vec = Select(title="Bairro:", value=bairro_init, options=bairros)
#callback
def update_data(attrname, old, new):
    vec_val = select_vec.value
    d_aux = data_cluster.loc[data_cluster.location==vec_val]
    d_aux["x"] = d_aux.index.values
    source2.data = dict(d_aux)
    
#pra fazer as actualizações
for w in [select_vec]:
    w.on_change('value', update_data)
    
inputs = column([select_vec], width=150)
vazio = row(width=159)
row_1 = row([inputs, p1])
row_2 = row([vazio, p2])

curdoc().add_root(row(gridplot([[row_1],[row_2]]), width=400))
