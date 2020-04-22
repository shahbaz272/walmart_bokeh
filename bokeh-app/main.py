#!/usr/bin/env python
# coding: utf-8

# In[1]:


from bokeh.layouts import column,widgetbox,row
from bokeh.models import ColumnDataSource, Slider, HoverTool,CrosshairTool,LinearColorMapper, TapTool
from bokeh.plotting import figure
from bokeh.themes import Theme
from bokeh.io import show, output_notebook,curdoc
from bokeh.models import DatetimeTickFormatter,Div,NumeralTickFormatter
from bokeh.models.tickers import DaysTicker,SingleIntervalTicker

from math import pi

from os.path import join, dirname


import matplotlib.pyplot as plt

import pandas as pd
import numpy as np


# In[2]:


#output_notebook()


# In[3]:


def error(preds,org,weights):
    error = np.mean(weights*np.abs(preds-org))
    return error


# In[4]:


all_preds = pd.read_csv(join(dirname(__file__), 'data/walmart_sales_prediction_bokeh.csv'),parse_dates = [0])
    
#all_preds = pd.read_csv('walmart_sales_prediction_bokeh.csv',parse_dates=[0])


# In[5]:


sb_dates = pd.to_datetime(['2010-02-12','2011-02-11','2012-02-10','2013-02-08'])
labor_dates = pd.to_datetime(['2010-09-10','2011-09-09','2012-09-07','2013-09-06'])
thanksgiving_dates = pd.to_datetime(['2010-11-26','2011-11-25','2012-11-23','2013-11-29'])
christmas_dates = pd.to_datetime(['2010-12-31','2011-12-30','2012-12-28','2013-12-27'])



all_preds['Which_Holiday'] = 'NoHoliday'
all_preds.loc[(np.isin(all_preds[['Date']].values, sb_dates.values).reshape(-1)),'Which_Holiday'] = 'Super_Bowl'
all_preds.loc[(np.isin(all_preds[['Date']].values, labor_dates.values).reshape(-1)),'Which_Holiday'] = 'Labor_Day'
all_preds.loc[(np.isin(all_preds[['Date']].values, thanksgiving_dates.values).reshape(-1)),'Which_Holiday'] = 'Thanksgiving'
all_preds.loc[(np.isin(all_preds[['Date']].values, christmas_dates.values).reshape(-1)),'Which_Holiday'] = 'Christmas'


# In[6]:


ap = all_preds.set_index(['Store','Dept'])
train_ap = ap.loc[ap.type=='Train']
test_ap = ap.loc[ap.type=='Test']


# In[7]:


error_table = pd.DataFrame(train_ap.reset_index().groupby(['Store','Dept']).apply(lambda x: error(x['preds'],x['Weekly_Sales'],x['IsHoliday'])),columns = ['WMSE'])


# In[8]:


error_mat = pd.DataFrame(train_ap.reset_index().groupby(['Dept','Store']).apply(lambda x: error(x['preds'],x['Weekly_Sales'],x['IsHoliday'])),columns = ['WMSE']).reset_index()
error_mat = error_mat.astype({'Dept':int,'Store':int})


# In[15]:


#def bkapp(doc):

#Define data sources
s = 1
d = 1
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

source_error = ColumnDataSource(data = {
    'Store' : error_mat.Store,
    'Dept' : error_mat.Dept,
    'error' : error_mat.WMSE
})


#Define plot ranges    
sales_min = min(0,min(ap.loc[s,d].Weekly_Sales),min(ap.loc[s,d].preds))
sales_max = max(max(ap.loc[s,d].Weekly_Sales),max(ap.loc[s,d].preds))
sales_range = sales_max-sales_min

#Define plot properties
plot = figure(x_axis_label = 'Date (MM/YYYY)',
              y_axis_label='Sales',
              x_axis_type='datetime',
              plot_width=800,
              plot_height=400,
              y_range = (sales_min,sales_max+sales_range*0.1),
              title = "Sales predictions for Walmart Store %d, Department %d" % (s,d)
             )
plot.yaxis[0].formatter = NumeralTickFormatter(format="$ 0.00 a")
plot.xaxis.ticker = DaysTicker(days=[1])
plot.xaxis.major_label_orientation = pi/4

error = round(error_table.loc[s,d][0],2)
error_text_place.text = '<b>(Weighted Mean Absolute Error) WMAE = ' + str(error)

# Define lines
preds_line_train = plot.line(x= 'Date',
          y='preds',
          alpha=0.8,
          line_color='blue',
          line_width=2,
          source=source_train,
          hover_alpha=0.5,
          legend_label='Predictions',muted_alpha=0.2)

preds_line_test = plot.line(x= 'Date',
          y='preds',
          alpha=0.8,
          line_color='green',
          line_width=2,
          source=source_test,
          legend_label='Predictions',muted_alpha=0.2)


sales_line = plot.line(x='Date',
          y='Sales',
          source=source_train,
          alpha=0.8,
          line_color='red',
          hover_color='red',
          hover_alpha=0.5,
          line_width=2,
          legend_label='Actual Sales',muted_alpha=0.2)

#Define Holiday markers      
holiday_circles_train = plot.circle(x='Date',
                                    y='IsHoliday',
                                    source=source_train,
                                    legend_label = 'Holiday',
                                    color='black',
                                    size=5,muted_alpha=0.2)

holiday_circles_test = plot.circle(x='Date',
                                   y='IsHoliday',
                                   source=source_test,
                                   legend_label = 'Holiday',
                                   color='black',
                                   size=5,muted_alpha=0.2)

#Define HoverTool    
hover_sales_train = HoverTool(tooltips = [('Date','@Date{%Y-%m-%d}'),
                             ('Holiday','@Which_Holiday'),
                             ('Sales','@Sales{$ 0.00 a}'),
                            ('preds','@preds{$ 0.00 a}')], mode='vline',formatters={'@Date':'datetime'}
                 ,renderers = [sales_line])

hover_sales_test = HoverTool(tooltips = [('Date','@Date{%Y-%m-%d}'),
                             ('Holiday','@Which_Holiday'),
                            ('preds','@preds{$ 0.00 a}')], mode='vline',formatters={'@Date':'datetime'}
                 ,renderers = [preds_line_test])

plot.add_tools(hover_sales_train,hover_sales_test)
plot.add_tools(CrosshairTool())
plot.legend.click_policy="mute"



 #Heatmap
hm_mapper = LinearColorMapper(palette='Magma256', low=error_mat.WMSE.min(), high=error_mat.WMSE.max())

hm = figure(title="Error heatmap",
           x_axis_label='Department', y_axis_label='Store',
           plot_width=800, plot_height=400,
           toolbar_location='below'
           )
hm_hover = HoverTool(tooltips = [('Store','@Store'),
                                ('Department','@Dept'),
                                ('Error','@error{0.00}')])

hm.grid.grid_line_color = None
#hm.yaxis.ticker = SingleIntervalTicker(interval=1)
hm.axis.major_tick_line_color = None
#hm.xaxis.major_label_orientation = pi/2

hm.rect(x="Dept", y="Store", width=1, height=1,
   source=source_error,
   fill_color={'field': 'error', 'transform': hm_mapper},
   line_color=None,hover_line_color = 'white',
       hover_color = {'field': 'error', 'transform': hm_mapper})

hm.add_tools(hm_hover)


#####
tap = TapTool()
hm.add_tools(tap)

def tap_callback(attr,old,new):

    try:
        selected_index = source_error.selected.indices[0]
        s = selected_store = int(error_mat.iloc[selected_index]['Store'])
        d = selected_dept = int(error_mat.iloc[selected_index]['Dept'])

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

        error_text_place.text = '<b>(Weighted Mean Absolute Error) WMAE = ' + str(round(error_table.loc[s,d][0],2))


        sales_min = min(0,min(ap.loc[s,d].Weekly_Sales),min(ap.loc[s,d].preds))
        sales_max = max(max(ap.loc[s,d].Weekly_Sales),max(ap.loc[s,d].preds))
        sales_range = sales_max-sales_min

        plot.y_range.start = sales_min
        plot.y_range.end = sales_max + sales_range*0.1

        plot.title.text = "Sales predictions for Walmart Store %d, Department %d" % (s,d)

    except IndexError:
        pass

source_error.selected.on_change('indices',tap_callback)
####


header = Div(text = '<b>Walmart Sales Forecasting', style = {'font-size':'200%','color':'blue'})

readme1 = Div(text = 'Click on the heatmap to see the sales and forecast for the Store-Department pair.<br>Tap the legend to <i>mute</i> lines',
             style = {'font-size':'150%'})


#Layout   
layout = column(header,readme1,error_text_place,plot,hm)

#    doc.add_root(layout)
curdoc().add_root(layout)
curdoc().title = 'Walmart Sales Forecasting'


# In[16]:


#show(bkapp)


# In[ ]:





# In[ ]:





# In[ ]:




