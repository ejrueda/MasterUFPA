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
from bokeh.models.widgets import Slider, TextInput
from bokeh.transform import cumsum
from bokeh.palettes import Category10

reset_output()
output_notebook()
#nombres de los vecindarios
names = ["Palace Hills", "Northwest","Old Town", "Safe Town","Southwest","Downtown",
         "Wilson Forest", "Scenic Vista", "Broadview", "Chapparal", "TerrapinSprings",
         "Pepper Mill", "CheddardFord", "Easton", "Weston", "Southon", "Oak Willow",
         "East Parton", "West Parton"]
#se cargan los datasets
data = pd.read_csv("boxplot/data.csv",parse_dates=True,infer_datetime_format=True, index_col=0)
df_bxplt = pd.read_csv("boxplot/data_boxplot.csv")
df_outs = pd.read_csv("boxplot/df_outs.csv")

x_ticks = ["sewer_and_water","power","roads_and_bridges","medical","buildings","shake_intensity"]
p = figure(plot_height=400, plot_width=800, title="Boxplot",background_fill_color="#efefef",
           tools="box_select,pan,wheel_zoom,box_zoom,reset,save", x_range=x_ticks, y_range=(-1,11))

p2 = figure(plot_height=300, plot_width=300, title="quantidade de outliers",
           tools="hover,box_select,pan,wheel_zoom,box_zoom,reset,save", x_range=x_ticks)
p2.xaxis.major_label_orientation = "vertical"

v_day = 6 #Configuración inicial del dia
v_loc = 1 #Configuración inicial del vecindario
y_rupper = df_bxplt.loc[df_bxplt.day==v_day].loc[df_bxplt.location==v_loc]["upper"].values
y_rlower = df_bxplt.loc[df_bxplt.day==v_day].loc[df_bxplt.location==v_loc]["lower"].values
y_q1 = df_bxplt.loc[df_bxplt.day==v_day].loc[df_bxplt.location==v_loc]["q1"].values
y_q2 = df_bxplt.loc[df_bxplt.day==v_day].loc[df_bxplt.location==v_loc]["q2"].values
y_q3 = df_bxplt.loc[df_bxplt.day==v_day].loc[df_bxplt.location==v_loc]["q3"].values
source_seg = ColumnDataSource(dict(x=x_ticks, y_lower=y_rlower, y_upper=y_rupper, q1=y_q1, q2=y_q2, q3=y_q3))
#Para graficar las lineas superiores del boxplot
fig_rect1 = p.rect("x", "y_upper", 0.2, 0.01, line_color="black", source=source_seg)
#Para graficar las lineas inferiores del boxplot
fig_rect2 = p.rect("x", "y_lower", 0.2, 0.01, line_color="black", source=source_seg)
#Para graficar los segmentos del boxplot
fig_seg1 = p.segment("x", "y_lower", "x", "q1", line_color="black", source=source_seg)
fig_seg2 = p.segment("x", "y_upper", "x", "q3", line_color="black", source=source_seg)
#Para graficar las barras
fig_bar1 = p.vbar("x", 0.3, "q1", "q2", fill_color="#3B8686", line_color="black", source=source_seg, legend="q1 até q2")
fig_bar2 = p.vbar("x", 0.3, "q2", "q3", fill_color="#E08E79", line_color="black", source=source_seg, legend="q2 até q3")
p.add_tools(HoverTool(renderers=[fig_bar1, fig_bar2,
                                fig_rect1, fig_rect2,
                                fig_seg1, fig_seg2], tooltips=[("quartil 1","@q1"),
                                                                ("quartil 2","@q2"),
                                                                ("quartil 3","@q3"),
                                                                ("min", "@y_lower"),
                                                                ("max","@y_upper")]))
#Para graficar los outliers si es que hay
#para pintar solo de a un punto si se presentan los mismos datos
aux_data = df_outs.groupby(["day","vecindario","x_outs","y_outs"],as_index=False).last().reset_index(drop=True)

source_out = ColumnDataSource(dict(aux_data[aux_data.day==v_day].loc[aux_data[aux_data.day==v_day].vecindario==v_loc]))
fig_out = p.circle("x_outs", "y_outs", size=6, color="#F38630", fill_alpha=0.6, source=source_out, legend="outliers")
#Para insertar el Hover solo a esos datos
p.add_tools(HoverTool(renderers=[fig_out], tooltips=[("value","@y_outs")]))
p.ygrid.grid_line_color = "white"
p.xgrid.grid_line_color = None

#PARA EL GRÁFICO DE BARRAS QUE LLEVA LA CANTIDAD DE OUTLIERS POR CADA TIPO
num_outs = []
df_outs_aux = df_outs.loc[df_outs.vecindario==1].loc[df_outs.loc[df_outs.vecindario==1].day==6]
for t in x_ticks:
    num_outs.append(df_outs_aux.loc[df_outs_aux.x_outs==t].shape[0])
source_nout = ColumnDataSource(dict(x_outs=x_ticks, y_outs=num_outs))
p2.vbar("x_outs", 0.3, "y_outs", fill_color="#3B8686", line_color="black", source=source_nout)
p2.hover.tooltips = [("quantidade","@y_outs")]
#Barras de interacción
s_day_text = "registro de acordo com o dia"
s_day = Slider(title=s_day_text, value=6.0, start=6.0, end=11.0, step=1.0)
s_vec_text = "vecindario"
s_vec= Slider(title=s_vec_text, value=1.0, start=1.0, end=19.0, step=1.0)

def update_data(attrname, old, new):
    day_value = s_day.value
    vec_value = s_vec.value
    y_rupper = df_bxplt.loc[df_bxplt.day==day_value].loc[df_bxplt.location==vec_value]["upper"].values
    y_rlower = df_bxplt.loc[df_bxplt.day==day_value].loc[df_bxplt.location==vec_value]["lower"].values
    y_q1 = df_bxplt.loc[df_bxplt.day==day_value].loc[df_bxplt.location==vec_value]["q1"].values
    y_q2 = df_bxplt.loc[df_bxplt.day==day_value].loc[df_bxplt.location==vec_value]["q2"].values
    y_q3 = df_bxplt.loc[df_bxplt.day==day_value].loc[df_bxplt.location==vec_value]["q3"].values
    df_aux = pd.DataFrame()
    df_aux["x"] = x_ticks
    df_aux["y_lower"] = y_rlower
    df_aux["y_upper"] = y_rupper
    df_aux["q1"] = y_q1
    df_aux["q2"] = y_q2
    df_aux["q3"] = y_q3
    source_seg.data = dict(df_aux)
    #Para actualizar los datos de los outliers
    source_out.data = dict(aux_data[aux_data.day==day_value].loc[aux_data[aux_data.day==day_value].vecindario==vec_value])
    #Para actualizar el gráfico de barras
    num_outs = []
    df_outs_aux = df_outs.loc[df_outs.vecindario==vec_value].loc[df_outs.loc[df_outs.vecindario==vec_value].day==day_value]
    for t in x_ticks:
        num_outs.append(df_outs_aux.loc[df_outs_aux.x_outs==t].shape[0])
    source_nout.data = dict(x_outs=x_ticks, y_outs=num_outs)
    
    
for w in [s_day, s_vec]:
    w.on_change('value', update_data)

# Set up layouts and add to document
inputs = column([s_day,s_vec], width=200)

curdoc().add_root(row(inputs, gridplot([[p, p2]]), width=400))
