#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from os.path import join,dirname

from bokeh.layouts import column,widgetbox,row
from bokeh.models import ColumnDataSource, Slider, HoverTool,CrosshairTool
from bokeh.plotting import figure
from bokeh.themes import Theme
from bokeh.io import show, output_notebook, curdoc
from bokeh.models import DatetimeTickFormatter,Div,NumeralTickFormatter
from bokeh.models.tickers import DaysTicker
from math import pi

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np


# In[ ]:


def error(preds,org,weights):
    error = np.mean(weights*np.abs(preds-org))
    return error


# In[ ]:


all_preds = pd.read_csv(join(dirname(__file__), 'data/walmart_sales_prediction_bokeh.csv'),parse_dates = [0])


# In[ ]:





# In[ ]:


sb_dates = pd.to_datetime(['2010-02-12','2011-02-11','2012-02-10','2013-02-08'])
labor_dates = pd.to_datetime(['2010-09-10','2011-09-09','2012-09-07','2013-09-06'])
thanksgiving_dates = pd.to_datetime(['2010-11-26','2011-11-25','2012-11-23','2013-11-29'])
christmas_dates = pd.to_datetime(['2010-12-31','2011-12-30','2012-12-28','2013-12-27'])



all_preds['Which_Holiday'] = 'NoHoliday'
all_preds.loc[(np.isin(all_preds[['Date']].values, sb_dates.values).reshape(-1)),'Which_Holiday'] = 'Super_Bowl'
all_preds.loc[(np.isin(all_preds[['Date']].values, labor_dates.values).reshape(-1)),'Which_Holiday'] = 'Labor_Day'
all_preds.loc[(np.isin(all_preds[['Date']].values, thanksgiving_dates.values).reshape(-1)),'Which_Holiday'] = 'Thanksgiving'
all_preds.loc[(np.isin(all_preds[['Date']].values, christmas_dates.values).reshape(-1)),'Which_Holiday'] = 'Christmas'


# In[ ]:


ap = all_preds.set_index(['Store','Dept'])
train_ap = ap.loc[ap.type=='Train']
test_ap = ap.loc[ap.type=='Test']


# In[ ]:


total_error = error(train_ap.preds,train_ap.Weekly_Sales,train_ap.IsHoliday)


# In[ ]:


error_table = pd.DataFrame(train_ap.reset_index().groupby(['Store','Dept']).apply(lambda x: error(x['preds'],x['Weekly_Sales'],x['IsHoliday'])),columns = ['WMSE'])


# In[ ]:


s=1
d=1    


source_train = ColumnDataSource(data = {
    'Date': train_ap.loc[s,d].Date,
    'IsHoliday' : np.where(train_ap.loc[s,d].IsHoliday==True,train_ap.loc[s,d].Weekly_Sales,None),
    'Sales' : train_ap.loc[s,d].Weekly_Sales,
    'preds' : train_ap.loc[s,d].preds,
    'Which_Holiday' : train_ap.loc[s,d].Which_Holiday
})

source_test = ColumnDataSource(data = {
    'Date': test_ap.loc[s,d].Date,
    'IsHoliday' : np.where(test_ap.loc[s,d].IsHoliday==True,test_ap.loc[s,d].preds,None),
    'preds' : test_ap.loc[s,d].preds,
    'Which_Holiday' : test_ap.loc[s,d].Which_Holiday
})


sales_min = min(0,min(ap.loc[s,d].Weekly_Sales),min(ap.loc[s,d].preds))
sales_max = max(max(ap.loc[s,d].Weekly_Sales),max(ap.loc[s,d].preds))
sales_range = sales_max-sales_min

# Create plots and widgets
plot = figure(x_axis_label = 'Date (MM/YYYY)',
              y_axis_label='Sales',
              x_axis_type='datetime',
              plot_width=800,
              plot_height=400,
              y_range = (sales_min,sales_max+sales_range*0.1)
             )
plot.yaxis[0].formatter = NumeralTickFormatter(format="$ 0.00 a")
plot.xaxis.ticker = DaysTicker(days=[1])
plot.xaxis.major_label_orientation = pi/4

preds_line_train = plot.line(x= 'Date',
          y='preds',
          alpha=0.8,
          line_color='blue',
          line_width=2,
          source=source_train,
          hover_alpha=0.5,
          legend='Predictions',)

preds_line_test = plot.line(x= 'Date',
          y='preds',
          alpha=0.8,
          line_color='blue',
          line_width=2,
          source=source_test,
          hover_alpha=0.8,
          legend='Predictions',)

sales_line = plot.line(x='Date',
          y='Sales',
          source=source_train,
          alpha=0.8,
          line_color='red',
          hover_color='red',
          hover_alpha=0.5,
          line_width=2,
          legend='Actual Sales')

holiday_circles_train = plot.circle(x='Date',
                                    y='IsHoliday',
                                    source=source_train,
                                    legend = 'Holiday',
                                    color='black',
                                    size=5)

holiday_circles_test = plot.circle(x='Date',
                                   y='IsHoliday',
                                   source=source_test,
                                   legend = 'Holiday',
                                   color='black',
                                   size=5)




hover_sales_train = HoverTool(tooltips = [('Date','@Date{%Y-%m-%d}'),
                             ('IsHoliday','@Which_Holiday'),
                             ('Sales','@Sales{$ 0.00 a}'),
                            ('preds','@preds{$ 0.00 a}')], mode='vline',formatters={'@Date':'datetime'}
                 ,renderers = [sales_line])

hover_sales_test = HoverTool(tooltips = [('Date','@Date{%Y-%m-%d}'),
                             ('IsHoliday','@Which_Holiday'),
                            ('preds','@preds{$ 0.00 a}')], mode='vline',formatters={'@Date':'datetime'}
                 ,renderers = [preds_line_test])

plot.add_tools(hover_sales_train,hover_sales_test)
plot.add_tools(CrosshairTool())

slider_store = Slider(start=1, end=45, value=1, step=1, title='Store')
slider_dept = Slider(start=1,end=99,value=1,step=1,title='Dept')

error = round(error_table.loc[s,d][0],2)
error_text = Div(text='WMAE = '+ str(error))


def callback_slider(attr, old, new):
    s = slider_store.value
    d = slider_dept.value

    new_train_data = {
        'Date': train_ap.loc[s,d].Date,
        'IsHoliday' : np.where(train_ap.loc[s,d].IsHoliday==True,train_ap.loc[s,d].Weekly_Sales,None),
        'Sales' : train_ap.loc[s,d].Weekly_Sales,
        'preds' : train_ap.loc[s,d].preds,
        'Which_Holiday' : train_ap.loc[s,d].Which_Holiday
    }

    source_train.data = new_train_data

    new_test_data = {
        'Date': test_ap.loc[s,d].Date,
        'IsHoliday' : np.where(test_ap.loc[s,d].IsHoliday==True,test_ap.loc[s,d].preds,None),
        'preds' : test_ap.loc[s,d].preds,
        'Which_Holiday' : test_ap.loc[s,d].Which_Holiday
    }

    source_test.data = new_test_data

    error_text.text = 'WMAE = ' + str(round(error_table.loc[s,d][0],2))


    sales_min = min(0,min(ap.loc[s,d].Weekly_Sales),min(ap.loc[s,d].preds))
    sales_max = max(max(ap.loc[s,d].Weekly_Sales),max(ap.loc[s,d].preds))
    sales_range = sales_max-sales_min

    plot.y_range.start = sales_min
    plot.y_range.end = sales_max + sales_range*0.1
    
slider_store.on_change('value', callback_slider)
slider_dept.on_change('value',callback_slider)


layout = column(widgetbox(slider_store,slider_dept,error_text), plot)

curdoc().add_root(layout)
curdoc().title = 'Walmart Sales Forecasting'

