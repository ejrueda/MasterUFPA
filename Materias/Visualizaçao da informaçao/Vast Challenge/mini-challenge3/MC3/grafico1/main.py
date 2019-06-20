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
from bokeh.palettes import Category10, Category20
import nltk

#Se cargan los datos
data = pd.read_csv("YInt.csv",index_col=0, parse_dates=True, infer_datetime_format=True)
data = data.dropna()
def histogram(df, col, bins=30):
    edges = []
    hist = []
    largura = max(df.index) - min(df.index)
    h = largura/bins
    aux = min(df.index)
    for i in range(100):
        edges.append(aux)
        hist.append(df.loc[aux:aux + h].count()[col])
        aux = aux + h
    edges.append(aux)
    
    return edges, hist

c_power = ["power", "energy", "electric", "electrician", "electricity", "powerlines", "electric"]
c_shake = ["shake", "earthquake", "quake", "tremble", "rubble", "earthquakes"]
c_hospital = ["hospital", "hospitals", "die", "dead", "death"]
c_disaster = ["disaster", "disasters"]
c_sewer = ["sewer", "sewage"]
c_evacuation = ["evacuation", "evacuate"]
others = ["nuclear", "repair", "structutral", "rescue", "emergency", "dangerous", "problem", "cry",
          "apocalypse", "sad", "ambulance", "victims", "help", "tragedy", "destroy", "roads", "scar", "hazard",
          "bridge", "damage", "build", "radiation", "crew", "shudder", "incident", "wobble"]
p_usar = c_power + c_shake + c_hospital + c_disaster + c_sewer + c_evacuation + others
idx_data = []
loc_data = []
acc_data = []
msg_data = []
for idx, cor in data.iterrows():
    for word in p_usar:
        if word in cor.message:
            idx_data.append(idx)
            loc_data.append(cor.location)
            acc_data.append(cor.account)
            msg_data.append(cor.message)
            break

df_aux = pd.DataFrame(columns=["time","location","account","message"])
df_aux["time"] = idx_data
df_aux["location"] = loc_data
df_aux["account"] = acc_data
df_aux["message"] = msg_data
df_aux = df_aux.set_index("time")

#Distribução geral dos dados
p1 = figure(plot_height=300, plot_width=800, title="Quantidade de tweets por intervalo de tempo para todos os bairros",
           tools="hover,pan,wheel_zoom,box_zoom,reset,save",x_axis_type='datetime')
p1.xaxis.axis_label = "dias registrados"
p1.yaxis.axis_label = "quantidade de tweets"
#quantidade de tweets por bairro
p2 = figure(plot_height=300, plot_width=300, title="Quantidade de tweets por bairro",
           tools="hover,pan,wheel_zoom,box_zoom,reset,save",x_range=(-3.5,3.5), y_range=(-3.5,3.5))
#palavras mais faladas
p3 = figure(plot_height=300, plot_width=600, title="Quantidade de tweets por bairro",
           tools="hover,pan,wheel_zoom,box_zoom,reset,save",x_axis_type='datetime', x_range=p1.x_range)
p3.xaxis.axis_label = "dias registrados"
p3.yaxis.axis_label = "quantidade de tweets"
#hashtag mais frequentes
p4 = figure(plot_height=300, plot_width=300, title="Quantidade de usuarios por bairro",
           tools="hover,pan,wheel_zoom,box_zoom,reset,save",x_range=(-4,4), y_range=(-3.5,3.5))
names = ["Palace Hills", "Northwest", "Old Town", "Safe Town", "Southwest", "Downtown",
         "Wilson Forest", "Scenic Vista", "Broadview", "Chapparal", "Terrapin Springs",
         "Pepper Mill", "Cheddarford", "Easton", "Weston", "Southton", "Oak Willow",
         "East Parton", "West Parton"]
#Barras de interacción
#s_day = RangeSlider(title="intervalo de dias", start=6, end=10, value=(6, 10), step=1)
#s_range = RangeSlider(title="intervalo de horas", start=0, end=24, value=(0,24), step=1)
select_vec = Select(title="Bairro:", value="Palace Hills", options=names)
#Gráfico 1
edges1, hist1 = histogram(data, "location", bins=100)
source1 = ColumnDataSource(dict(hist=hist1,left=edges1[:-1], right=edges1[1:]))
p1.quad(top="hist", bottom=0, left="left", right="right", color = "blue",
            line_color="white", alpha=0.7, source=source1, legend="tweets")
p1.hover.tooltips = [("data inicial", "@left{%F %T}"),
                    ("data final", "@right{%F %T}"),
                    ("quantidade", "@hist")]
p1.hover.formatters = {'left': 'datetime', 'right': 'datetime'}
p1.hover.mode = "vline"

#Gráfico 2
edges2, hist2 = histogram(df_aux.loc[df_aux.location=="Palace Hills"], "location", bins=100)
source2 = ColumnDataSource(dict(hist=hist2,left=edges2[:-1], right=edges2[1:]))
p3.quad(top="hist", bottom=0, left="left", right="right", color = "red",
            line_color="white", alpha=0.7, source=source2, legend="tweets")
p3.hover.tooltips = [("data inicial", "@left{%F %T}"),
                    ("data final", "@right{%F %T}"),
                    ("quantidade", "@hist")]
p3.hover.formatters = {'left': 'datetime', 'right': 'datetime'}
p3.hover.mode = "vline"

#Gráfico 3
data_aux = data.copy()
data_aux.drop(["account","message"], axis=1)
data_aux["values"] = 1
data_aux = data_aux.groupby("location").sum()
data_aux = data_aux.drop(["UNKNOWN", "<Location with-held due to contract>"])
source3 = ColumnDataSource(dict(data_aux, angle=data_aux["values"]/(data_aux["values"]).sum() * 2*np.pi,
                                  color=Category20[19], location=data_aux.index))
p2.wedge(x=0, y=0, radius=3,start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
         line_color="black", fill_color='color', source=source3)
p2.hover.tooltips = [("bairro", "@location"),
                     ("quantidade", "@values")]

#Gráfico 4
data_aux2 = data.copy()
data_aux2.drop("message", axis=1)
data_aux2["values"] = 1
data_aux2 = data_aux2.groupby(["location", "account"], as_index=False).sum()
data_aux2 = data_aux2.set_index("location")
data_aux2 = data_aux2.drop(["UNKNOWN", "<Location with-held due to contract>"])
data_aux2["values"] = 1
data_aux2 = data_aux2.groupby("location").sum()
source4 = ColumnDataSource(dict(data_aux2, angle=data_aux["values"]/(data_aux["values"]).sum() * 2*np.pi,
                               color=Category20[19], location=data_aux2.index))
p4.wedge(x=0, y=0, radius=3,start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
         line_color="black", fill_color='color', source=source4)
p4.hover.tooltips = [("bairro", "@location"),
                     ("quantidade", "@values")]

def update_data(attrname, old, new):

    #Para obtener los valores actuales
    #day_val = s_day.value
    vec_val = select_vec.value
    #hr_val = s_range.value
    #Actualizar grafico 3
    edges2, hist2 = histogram(df_aux.loc[df_aux.location==vec_val], "location", bins=100)
    source2.data = dict(hist=hist2,left=edges2[:-1], right=edges2[1:])
    
    
#Para hacer las actualizaciones
for w in [select_vec]:
    w.on_change('value', update_data)

row_1 = row([select_vec, p1, p2], width=1100)
row_2 = row([p3])
row_3 = row([])
row_4 = row([])

curdoc().add_root(row(gridplot([[row_1],[row_2],[row_3],[row_4]]), width=400))