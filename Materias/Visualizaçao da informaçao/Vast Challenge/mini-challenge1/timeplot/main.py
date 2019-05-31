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

#reset_output()
#output_notebook()
#se cargan los datasets
data = pd.read_csv("timeplot/data.csv",parse_dates=True,infer_datetime_format=True, index_col=0)
p = figure(plot_height=400, plot_width=800, title="Reporte do danho no tempo",
           tools="hover,pan,wheel_zoom,box_zoom,reset,save", x_axis_type='datetime')

p1 = figure(plot_height=200, plot_width=300, title="Distribuição de Power",
           tools="hover,pan,wheel_zoom,box_zoom,reset,save")
p2 = figure(plot_height=200, plot_width=300, title="Distribuição de Medical",
           tools="hover,pan,wheel_zoom,box_zoom,reset,save")
p3 = figure(plot_height=200, plot_width=300, title="Distribuição de Sewer and water",
           tools="hover,pan,wheel_zoom,box_zoom,reset,save")
p4 = figure(plot_height=200, plot_width=300, title="Distribuição de Roads and bridges",
           tools="hover,pan,wheel_zoom,box_zoom,reset,save")
p5 = figure(plot_height=200, plot_width=300, title="Distribuição de Buildings",
           tools="hover,pan,wheel_zoom,box_zoom,reset,save")
p6 = figure(plot_height=200, plot_width=300, title="Distribuição de Shake intensity",
           tools="hover,pan,wheel_zoom,box_zoom,reset,save")
#Barras de interacción
s_day = Slider(title="registro de acordo com o dia", value=6.0, start=6.0, end=11.0, step=1.0)
s_vec= Slider(title="bairro", value=1.0, start=1.0, end=19.0, step=1.0)
a = datetime.datetime(2020, 3, 22, hour=1, minute=2)
b = datetime.datetime(2020, 3, 26, hour=11, minute=2)
s_range = RangeSlider(title="intervalo de horas", start=0, end=24, value=(0,24), step=1)

data_aux = data.loc[data.index.day==6].loc[data.loc[data.index.day==6].location==1]
source = ColumnDataSource(dict(data_aux, x=data_aux.index))
#fig_t1 = p.line(x="x", y="power", source=source)
fig_t1 = p.circle(x="x", y="power", source=source, legend="Power",color=Category10[6][0])
fig_t2 = p.circle(x="x", y="medical", source=source, legend="Medical", color=Category10[6][1])
fig_t3 = p.circle(x="x", y="sewer_and_water", source=source, legend="Sewer and water", color=Category10[6][2])
#roads_and_bridges
fig_t4 = p.circle(x="x", y="roads_and_bridges", source=source, legend="Roads and bridges", color=Category10[6][3])
#buildings
fig_t5 = p.circle(x="x", y="buildings", source=source, legend="Buildings", color=Category10[6][4])
#shake_intensity
fig_t5 = p.circle(x="x", y="shake_intensity", source=source, legend="Shake intensity", color=Category10[6][5])
p.legend.click_policy="hide"
#p.add_tools(HoverTool(renderers=[fig_t1], tooltips=[("magnitude","$y")]))
p.hover.tooltips = [("magnitude", "$y")]

#PARA DIBUJAR LA DISTRIBUICIÓN DE LOS DATOS
types = ["power","medical","sewer_and_water","roads_and_bridges","buildings","shake_intensity"]
Legend = ["Power","Medical","Sewer and water","Roads and bridges","Buildings","Shake_intensity"]

hist1, edges1 = np.histogram(data_aux["power"], bins=20)
hist2, edges2 = np.histogram(data_aux["medical"], bins=20)
hist3, edges3 = np.histogram(data_aux["sewer_and_water"], bins=20)
hist4, edges4 = np.histogram(data_aux["roads_and_bridges"], bins=20)
hist5, edges5 = np.histogram(data_aux["buildings"], bins=20)
hist6, edges6 = np.histogram(data_aux["shake_intensity"], bins=20)
source1 = ColumnDataSource(dict(hist=hist1,left=edges1[:-1], right=edges1[1:]))
source2 = ColumnDataSource(dict(hist=hist2,left=edges2[:-1], right=edges2[1:]))
source3 = ColumnDataSource(dict(hist=hist3,left=edges3[:-1], right=edges3[1:]))
source4 = ColumnDataSource(dict(hist=hist4,left=edges4[:-1], right=edges4[1:]))
source5 = ColumnDataSource(dict(hist=hist5,left=edges5[:-1], right=edges5[1:]))
source6 = ColumnDataSource(dict(hist=hist6,left=edges6[:-1], right=edges6[1:]))
p1.quad(top="hist", bottom=0, left="left", right="right", color = Category10[6][0],
            line_color="white", alpha=0.7, source=source1)
p1.hover.tooltips = [("min","@left"), ("max","@right"), ("quantidade","@hist")]
p2.quad(top="hist", bottom=0, left="left", right="right", color = Category10[6][1],
            line_color="white", alpha=0.7, source=source2)
p2.hover.tooltips = [("min","@left"), ("max","@right"), ("quantidade","@hist")]
p3.quad(top="hist", bottom=0, left="left", right="right", color = Category10[6][2],
           line_color="white", alpha=0.7, source=source3)
p3.hover.tooltips = [("min","@left"), ("max","@right"), ("quantidade","@hist")]
p4.quad(top="hist", bottom=0, left="left", right="right", color = Category10[6][3],
            line_color="white", alpha=0.7, source=source4)
p4.hover.tooltips = [("min","@left"), ("max","@right"), ("quantidade","@hist")]
p5.quad(top="hist", bottom=0, left="left", right="right", color = Category10[6][4],
            line_color="white", alpha=0.7, source=source5)
p5.hover.tooltips = [("min","@left"), ("max","@right"), ("quantidade","@hist")]
p6.quad(top="hist", bottom=0, left="left", right="right", color = Category10[6][5],
            line_color="white", alpha=0.7, source=source6)
p6.hover.tooltips = [("min","@left"), ("max","@right"), ("quantidade","@hist")]
#Para actualizar los datos con el cambio de los widgets

def update_data(attrname, old, new):
    day_value = s_day.value
    vec_value = s_vec.value
    range_value = s_range.value
    data_aux = data.loc[data.index.day==day_value].loc[data.loc[data.index.day==day_value].location==vec_value]
    data_aux = data_aux.loc[data_aux.index.hour>=range_value[0]].loc[data_aux.loc[data_aux.index.hour>=range_value[0]].index.hour<=range_value[1]]
    source.data = dict(data_aux, x=data_aux.index)
    #Para actualizar los datos de las distribuiciones
    hist1, edges1 = np.histogram(data_aux["power"], bins=20)
    hist2, edges2 = np.histogram(data_aux["medical"], bins=20)
    hist3, edges3 = np.histogram(data_aux["sewer_and_water"], bins=20)
    hist4, edges4 = np.histogram(data_aux["roads_and_bridges"], bins=20)
    hist5, edges5 = np.histogram(data_aux["buildings"], bins=20)
    hist6, edges6 = np.histogram(data_aux["shake_intensity"], bins=20)
    source1.data = dict(hist=hist1,left=edges1[:-1], right=edges1[1:])
    source2.data = dict(hist=hist2,left=edges2[:-1], right=edges2[1:])
    source3.data = dict(hist=hist3,left=edges3[:-1], right=edges3[1:])
    source4.data = dict(hist=hist4,left=edges4[:-1], right=edges4[1:])
    source5.data = dict(hist=hist5,left=edges5[:-1], right=edges5[1:])
    source6.data = dict(hist=hist6,left=edges6[:-1], right=edges6[1:])

for w in [s_vec,s_day,s_range]:
    w.on_change('value', update_data)

# Set up layouts and add to document
inputs = column([s_vec,s_day,s_range], width=200)
col1_1 = column([inputs], width=200)
col1_2 = column([p], width=800)
col1_3 = column([p2,p3], width=300)
fila_1 = row([col1_1, col1_2, col1_3])
fila_2 = row([p1,p4,p5,p6])

curdoc().add_root(row(gridplot([[fila_1],[fila_2]]), width=400))
