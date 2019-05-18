import pandas as pd
import numpy as np
from bokeh.plotting import figure, show, output_file
from bokeh.io import output_notebook, reset_output, curdoc
from bokeh.embed import file_html
from bokeh.models import LabelSet, Label, ColumnDataSource, FactorRange
from bokeh.transform import factor_cmap
import bokeh.plotting as bp
from bokeh.layouts import row, column
from bokeh.models.widgets import Slider, TextInput
from bokeh.transform import cumsum
from bokeh.palettes import Category10

#reset_output()
#output_notebook()
df_circle = pd.read_csv("./df_circle.csv")
df_circle2 = pd.read_csv("./df_circle2.csv")
p = figure(plot_height=400, plot_width=700, title="Diagrama de pizza",
              tools="pan,reset,save,wheel_zoom, hover")
source = ColumnDataSource(dict(df_circle))
source2 = ColumnDataSource(dict(df_circle2))

p.wedge(x=0, y=1, radius=0.2 ,start_angle=cumsum('angle', include_zero=True),
                end_angle=cumsum('angle'), line_color="black", fill_color='color', source=source,
                legend="type", fill_alpha=.8)

p.wedge(x=1, y=1, radius=0.2 ,start_angle=cumsum('angle', include_zero=True),
                end_angle=cumsum('angle'), line_color="black", fill_color='color', source=source2,
                legend="type", fill_alpha=.8)

s_day = Slider(title="dia", value=6.0, start=6.0, end=11.0, step=1.0)
magnitude = Slider(title="magnitude", value=1.0, start=0.0, end=10.0, step=0.1)

def select_data():
    day_value = s_day.value
    magnitude_val = magnitude.value
    selected = df_circle2[df_circle2.loc[:,'dval'] == day_value]
    
    return selected


def update_data(attrname, old, new):

    # Get the current slider values
    day_value = s_day.value
    df = select_data()
    source.data = dict(df)
    source2.data = dict(df_circle[df_circle.loc[:,'value'] == day_value])
    
for w in [s_day]:
    w.on_change('value', update_data)


# Set up layouts and add to document
inputs = column(s_day)

curdoc().add_root(row(inputs, p, width=800))
curdoc().title = "Sliders"
