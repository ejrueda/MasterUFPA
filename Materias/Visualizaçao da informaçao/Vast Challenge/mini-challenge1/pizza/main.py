import pandas as pd
import numpy as np
from bokeh.plotting import figure, show, output_file
from bokeh.io import output_notebook, reset_output, curdoc
from bokeh.embed import file_html
from bokeh.models import LabelSet, Label, ColumnDataSource, FactorRange
from bokeh.transform import factor_cmap
import bokeh.plotting as bp
from bokeh.layouts import row, column,layout, gridplot
from bokeh.models.widgets import Slider, TextInput
from bokeh.transform import cumsum
from bokeh.palettes import Category10, Category20


p = figure(plot_height=480, plot_width=800, title="Danho do terremoto por dias",
              tools="pan,reset,save,wheel_zoom, hover",x_range=(30,180), y_range=(35,185))
p2 = figure(plot_height=300, plot_width=300, title="Número de chamadas por vecindario",
              tools="pan,reset,save,wheel_zoom, hover",x_range=(-4,4), y_range=(-4,4))

p3 = figure(plot_height=300, plot_width=300, title="Intensidade do terremoto por vecindario",
              tools="pan,reset,save,wheel_zoom, hover",x_range=(-4,4), y_range=(-4,4))

df_pizza = pd.read_csv("pizza/df_pizza.csv")
df_acu = pd.read_csv("pizza/data_acu.csv")
df_shake_intensity = pd.read_csv("pizza/shake_intensity.csv")
p.image_url(url=['pizza/static/mapa.svg'], x=50,y=180,w=100,h=150)  

x = np.array([65,80,100,118,80,78,148,133,110,125,135,135,120,99,88,88,110,110,98])
y = np.array([135,145,153,130,100,122,98,53,60,68,70,95,98,125,125,110,85,105,105])
l_DS = []
#CREAR TODOS LOS DATASOURCE
default_day = 6
for k in range(1,20): 
    df_p = df_pizza.loc[df_pizza["location"]==k].loc[df_pizza.loc[df_pizza["location"]==k].day==default_day]
    df_p["angle"] = df_p["values"]/(df_p["values"]).sum() * 2*np.pi
    l_DS.append(ColumnDataSource(dict(df_p, color=Category10[6])))
i=0

for idx in range(0,19):
    if idx == 0:
        p.wedge(x=x[i], y=y[i], radius=3,start_angle=cumsum('angle', include_zero=True),
                end_angle=cumsum('angle'), line_color="black", fill_color='color', source=l_DS[idx],
                legend="type", fill_alpha=.8)
    else:
        p.wedge(x=x[i], y=y[i], radius=3,start_angle=cumsum('angle', include_zero=True),
                end_angle=cumsum('angle'), line_color="black", fill_color='color', source=l_DS[idx],
               fill_alpha=.8)
    p.hover.tooltips = [
        ("name", "@type"),
        ("magnitude", "@values")
    ]
    i += 1
#PINTAR LAS LLAMADAS POR VECINDARIO
df_source = df_acu.loc[df_acu["day"]==6]
source_acu = ColumnDataSource(dict(df_source, angle=df_source["values"]/(df_source["values"]).sum() * 2*np.pi,
                                  color=Category20[19]))

p2.wedge(x=0, y=0, radius=3,start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
         line_color="black", fill_color='color', source=source_acu,fill_alpha=1)
p2.hover.tooltips = [
        ("name", "@location"),
        ("chamadas", "@values"),
        ("dia", "@day")
    ]

#Pintar la intensidad del terremoto por vecindario
df_shake = df_shake_intensity.loc[df_shake_intensity["day"]==6]
source_shake = ColumnDataSource(dict(df_shake, angle=df_shake["values"]/(df_shake["values"]).sum() * 2*np.pi,
                                  color=Category20[19]))
p3.wedge(x=0, y=0, radius=3,start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
         line_color="black", fill_color='color', source=source_shake,fill_alpha=1)
p3.hover.tooltips = [
        ("name", "@location"),
        ("magnitude", "@values"),
        ("dia", "@day")
    ]

#Barras de interacción
s_day_text = "registro de acordo com o dia"
s_day = Slider(title=s_day_text, value=6.0, start=6.0, end=11.0, step=1.0)
#s_magnitude = Slider(title=s_day_text, value=6.0, start=6.0, end=11.0, step=1.0)

def update_data(attrname, old, new):

    # Get the current slider values
    day_value = s_day.value
    l_aux = []
    for i in range(0,19):
        df_p1 = df_pizza.loc[df_pizza["location"]==i+1].loc[df_pizza.loc[df_pizza["location"]==i+1].day==day_value]
        df_p1["angle"] = df_p1["values"]/(df_p1["values"]).sum() * 2*np.pi
        l_DS[i].data = dict(df_p1, color=Category10[6])
    
    #para actualizar el número de llamadas acumuladas  
    df_aux = df_acu.loc[df_acu["day"]==day_value]
    source_acu.data = dict(df_aux, angle=df_aux["values"]/(df_aux["values"]).sum() * 2*np.pi,
                                  color=Category20[19])
    
    df_aux2 = df_shake_intensity.loc[df_shake_intensity["day"]==day_value]
    source_shake.data = dict(df_aux2, angle=df_aux2["values"]/(df_aux2["values"]).sum() * 2*np.pi,
                                  color=Category20[19])
for w in [s_day]:
    w.on_change('value', update_data)

# Set up layouts and add to document
inputs = column(s_day, width=200)

curdoc().add_root(row(inputs, gridplot([[p]]), column(p2,p3), width=400))