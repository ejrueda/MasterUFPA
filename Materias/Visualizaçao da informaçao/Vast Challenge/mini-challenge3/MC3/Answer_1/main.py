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
from bokeh.transform import cumsum
from bokeh.palettes import Category10
import nltk

#Distribução geral dos dados
p1 = figure(plot_height=400, plot_width=800, title="Distribuição dos dados",
           tools="hover,pan,wheel_zoom,box_zoom,reset,save", x_axis_type='datetime')
#quantidade de tweets por bairro
p2 = figure(plot_height=400, plot_width=300, title="Quantidade do tweets por bairro",
           tools="hover,pan,wheel_zoom,box_zoom,reset,save")
#palavras mais faladas
p3 = figure(plot_height=400, plot_width=800, title="Palavras mais faladas",
           tools="hover,pan,wheel_zoom,box_zoom,reset,save")
#hashtag mais frequentes
p4 = figure(plot_height=400, plot_width=300, title="Hashtag top",
           tools="hover,pan,wheel_zoom,box_zoom,reset,save")

s_day = RangeSlider(title="intervalo de dias", start=6, end=10, value=(6, 10), step=1)
s_range = RangeSlider(title="intervalo de horas", start=0, end=24, value=(0,24), step=1)
names = ["Palace Hills", "Northwest","Old Town", "Safe Town","Southwest","Downtown",
         "Wilson Forest", "Scenic Vista", "Broadview", "Chapparal", "TerrapinSprings",
         "Pepper Mill", "CheddardFord", "Easton", "Weston", "Southon", "Oak Willow",
         "East Parton", "West Parton"]
select_vec = Select(title="Bairro:", value="foo", options=names)

col_1 = column([s_day,s_range,select_vec], width=200)
col_2 = column([p1, p3])
col_3 = column([p2, p4])

curdoc().add_root(row(gridplot([[col_1, col_2, col_3]]), width=400))