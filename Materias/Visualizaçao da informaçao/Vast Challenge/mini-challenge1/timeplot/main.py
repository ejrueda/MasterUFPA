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
from bokeh.models.widgets import Slider, TextInput, RangeSlider, DateRangeSlider
from bokeh.transform import cumsum
from bokeh.palettes import Category10
import datetime

reset_output()
output_notebook()
#se cargan los datasets
data = pd.read_csv("timeplot/data.csv",parse_dates=True,infer_datetime_format=True, index_col=0)
p = figure(plot_height=400, plot_width=800, title="Reporte do danho no tempo",
           tools="box_select,pan,wheel_zoom,box_zoom,reset,save")

p2 = figure(plot_height=300, plot_width=300, title="Distribuição dos dados",
           tools="hover,box_select,pan,wheel_zoom,box_zoom,reset,save")

#Barras de interacción
s_day = Slider(title="registro de acordo com o dia", value=6.0, start=6.0, end=11.0, step=1.0)
s_vec= Slider(title="vecindario", value=1.0, start=1.0, end=19.0, step=1.0)
a = datetime.datetime(2020, 3, 22, hour=1, minute=2)
b = datetime.datetime(2020, 3, 26, hour=11, minute=2)
s_range = RangeSlider(title="intervalo de horas", start=0, end=24, value=(0,24), step=1)

data_aux = data.loc[data.index.day==6].loc[data.loc[data.index.day==6].location==1]
source = ColumnDataSource(dict(data_aux, x=data_aux.index))
fig_t1 = p.line(x="x", y="power", source=source)

#Para actualizar los datos con el cambio de los widgets
def update_data(attrname, old, new):
    #data_aux = data.loc[data.index.day==6].loc[data.loc[data.index.day==6].location==1]
    #source = ColumnDataSource(dict(data_aux, x=data_aux.index))
    return None

for w in [s_vec,s_day,s_range]:
    w.on_change('value', update_data)

# Set up layouts and add to document
inputs = column([s_vec,s_day,s_range], width=200)

curdoc().add_root(row(inputs, gridplot([[p, p2]]), width=400))
