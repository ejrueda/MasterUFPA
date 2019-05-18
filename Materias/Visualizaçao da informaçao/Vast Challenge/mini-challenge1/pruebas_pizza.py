import pandas as pd
import numpy as np
from bokeh.plotting import figure, show, output_file
from bokeh.io import output_notebook, reset_output, curdoc
from bokeh.embed import file_html
from bokeh.models import LabelSet, Label, ColumnDataSource, FactorRange
from bokeh.transform import factor_cmap
import bokeh.plotting as bp
from bokeh.layouts import row, column,layout
from bokeh.models.widgets import Slider, TextInput
from bokeh.transform import cumsum
from bokeh.palettes import Category10


p = figure(plot_height=600, plot_width=1000, title="Diagrama de pizza",
              tools="pan,reset,save,wheel_zoom, hover",x_range=(30,180), y_range=(35,185))

df_pizza = pd.read_csv("df_pizza.csv")
output_file("test.html")
from os.path import dirname, join
url = join(dirname("mapa.svg"), 'logo.png')
p.image_url(url=url, x=50,y=180,w=100,h=150)  

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
        p.wedge(x=x[i], y=y[i], radius=1,start_angle=cumsum('angle', include_zero=True),
                end_angle=cumsum('angle'), line_color="black", fill_color='color', source=l_DS[idx],
                legend="type", fill_alpha=.8)
    else:
        p.wedge(x=x[i], y=y[i], radius=1,start_angle=cumsum('angle', include_zero=True),
                end_angle=cumsum('angle'), line_color="black", fill_color='color', source=l_DS[idx],
               fill_alpha=.8)
    p.hover.tooltips = [
        ("name", "@type"),
        ("# chamadas", "@calls"),
        ("magnitude", "@value")
    ]
    i += 1

s_day = Slider(title="dia do terremoto", value=6.0, start=6.0, end=11.0, step=1.0)
def update_data(attrname, old, new):

    # Get the current slider values
    day_value = s_day.value
    df_p1 = df_pizza.loc[df_pizza["location"]==1].loc[df_pizza.loc[df_pizza["location"]==1].day==day_value]
    df_p2 = df_pizza.loc[df_pizza["location"]==2].loc[df_pizza.loc[df_pizza["location"]==2].day==day_value]
    df_p1["angle"] = df_p1["values"]/(df_p1["values"]).sum() * 2*np.pi
    df_p2["angle"] = df_p2["values"]/(df_p2["values"]).sum() * 2*np.pi
    l_DS[0].data = dict(df_p1, color=Category10[6])
    l_DS[1].data = dict(df_p2, color=Category10[6])
    
for w in [s_day]:
    w.on_change('value', update_data)

# Set up layouts and add to document
inputs = column(s_day)

curdoc().add_root(row(inputs, p, width=400))
    