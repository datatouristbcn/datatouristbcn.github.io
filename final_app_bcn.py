import pandas as pd
import numpy as np
import math
import plotly.express as px
import plotly.graph_objects as go
import json

df=pd.read_csv('https://raw.githubusercontent.com/datatouristbcn/datatouristbcn.github.io/master/final_dataset_app.csv')
df['id_neighbourhood'] = df['id_neighbourhood'].apply(lambda x: '{0:0>2}'.format(x))
total_houses=pd.read_csv('https://raw.githubusercontent.com/datatouristbcn/datatouristbcn.github.io/master/total_houses.csv')

import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
with open('barris.geojson') as f:
    js = json.load(f)

fig = px.choropleth_mapbox(df[df.year==2019], geojson=js, locations='id_neighbourhood', featureidkey="properties.BARRI",hover_data=['neighbourhood'], color='Nominal ',
                           color_continuous_scale="OrRd",
                           range_color=(5, max(df[df.year==2019]['Nominal '])),
                           mapbox_style="carto-positron",
                           zoom=11, center = {"lat": 41.39, "lon": 2.15},
                           opacity=0.5,
                           labels={'variable':'unemployment rate'},
                           custom_data =['id_neighbourhood']
                          )
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":30})
fig.update_layout(annotations=[
        dict(
            x=0.5,
            y=1.0,
            showarrow=False,
            text='Custom annotation text',
            xref='paper',
            yref='paper'
        ),
    ])

xy=pd.DataFrame()
xy['neighbourhood']=df['id_neighbourhood'].unique()
x=list(df[(df.year==2019)&(df.variable=='Average weight of airbnb inside the neighbourhood')]['Nominal '])-(df[(df.year==2015)&(df.variable=='Average weight of airbnb inside the neighbourhood')]['Nominal '])
y=(list(df[(df.year==2019)&(df.variable=='Average renting price/m2')]['Nominal '])-df[(df.year==2015)&(df.variable=='Average renting price/m2')]['Nominal '])/df[(df.year==2015)&(df.variable=='Average renting price/m2')]['Nominal ']
xy['x']=x.values
xy['y']=y.values
xy['total_houses']=total_houses['total_houses'].values
xy['name']=df['neighbourhood'].unique()
fig_scatter = dict(data=#[go.Figure(data=
    [go.Scatter(x=xy['x'],#['Nominal '],
    y=xy['y'],#['Per_change_previous_year'],
    text=xy['name'],#['neighbourhood'],
    mode='markers',
    marker=dict(
        size=14,
        color=xy['total_houses'], 
        colorscale='Viridis',
        
        showscale=True
    )
            
)],layout=go.Layout(showlegend=False,xaxis= {'title':"Increase of Airbnb's presence(x100)%   (Color scale shows Living units density)" },yaxis={'title':"Increase in rent per m2(x100)%"},coloraxis_colorbar=dict(title="Color scale shows Living usinng density")
   ,annotations=[dict(xref='paper', yref='paper', x=0.0, y=1.05,
                               xanchor='left', yanchor='bottom',
                               text='Scatter plot showing the correlation between Increase in rent and Increase of Airbnb weight in the neighbourhood for the period 2015-2019' ,
                               
                               font=dict(family='Arial',
                                         size=14,
                                         color='rgb(37,37,37)'),
                               showarrow=False)
                                ]))




labels = ['Average renting price/m2', 'Average airbnb listings']
colors = ['rgb(255,128,0)','rgb(0,170,228)'] 

mode_size = [8, 8, 12, 8]
line_size = [2, 2, 4, 2]



fig_indexed= go.Figure()
df_indexed=df[df['id_neighbourhood']=='01']
for i in range(2):
        fig_indexed.add_trace(go.Scatter(x=df_indexed[df_indexed.variable==labels[i]]['year'], y=df_indexed[df_indexed.variable==labels[i]]['Indexed'], mode='lines',
            name=labels[i],
            line=dict(color=colors[i], width=line_size[i]),
            connectgaps=True,
             ))            
fig_indexed.update_layout(xaxis= {'nticks': 5})
#df = pd.read_csv('https://plotly.github.io/datasets/country_indicators.csv')
fig_indexed.update_layout(annotations=[
                            dict(xref='paper', yref='paper', x=0.0, y=1.05,
                              xanchor='left', yanchor='bottom',
                              text='Indexed graph in the period 2015-2019 for',
                              font=dict(family='Arial',
                                        size=16,
                                        color='rgb(37,37,37)'),
                              showarrow=False)
                                ])




available_indicators = df['variable'].unique()

app.layout = html.Div([
    html.Div([

        html.Div([
            dcc.Dropdown(
                id='crossfilter-xaxis-column',
                options=[{'label': i, 'value': i} for i in available_indicators],
                value='Average renting price/m2'
            ),
            dcc.RadioItems(
                id='crossfilter-xaxis-type',
                options=[{'label': i, 'value': i} for i in ['Nominal', '% change from previous year']],
                value='Nominal',
                labelStyle={'display': 'inline-block'}
            )
        ],
        style={'width': '40%', 'display': 'inline-block'}),


    ], style={
        'borderBottom': 'thin lightgrey solid',
        'backgroundColor': 'rgb(250, 250, 250)',
        'padding': '10px 5px'
    }),

    html.Div([
        dcc.Graph(
            id='crossfilter-indicator-scatter',
            figure=fig,
            hoverData={'points': [{'location': '01'}]}
        )
    ], style={'width': '49%', 'display': 'inline-block', 'padding': '0 10'}),

         
    
    html.Div([
        dcc.Graph(id='x-time-series'),
    ], style={'display': 'inline-block', 'width': '47%'}),
    html.Div(dcc.Slider(
        id='crossfilter-year--slider',
        min=df['year'].min(),
        max=df['year'].max(),
        value=df['year'].max(),
        marks={str(year): str(year) for year in df['year'].unique()},
        step=None
    ), style={'width': '49%', 'padding': '20px 20px 10px 20px'}),
    

    html.Div([
        dcc.Graph(id='scatter-regression',
        figure=fig_scatter
    )], style={'display': 'inline-block', 'width': '49%'}),
    html.Div([

        dcc.Graph(id='y-time-series',
                 figure=fig_indexed),
    ], style={'display': 'inline-block', 'width': '47%'})])#,#,


@app.callback(
    dash.dependencies.Output('crossfilter-indicator-scatter', 'figure'),
    [dash.dependencies.Input('crossfilter-xaxis-column', 'value'),

     dash.dependencies.Input('crossfilter-xaxis-type', 'value'),

     dash.dependencies.Input('crossfilter-year--slider', 'value')])

def update_graph(xaxis_column_name,
                 xaxis_type,
                 year_value):
    if xaxis_type == 'Nominal':
        val='Nominal '
    else: 
        val='Per_change_previous_year'
    dff = df[df['year'] == year_value]
    dff=dff[dff['variable']==xaxis_column_name]
    fig = px.choropleth_mapbox(dff, geojson=js, locations='id_neighbourhood', featureidkey="properties.BARRI",hover_data=['neighbourhood'], color=val,
                           color_continuous_scale="OrRd",
                           range_color=(min(dff[val]), max(dff[val])),
                           mapbox_style="carto-positron",
                           zoom=11,
                           center = {"lat": 41.39, "lon": 2.15},
                           opacity=0.5,
                           labels={'variable':'unemployment rate'},
                           custom_data =['id_neighbourhood']
                          )
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    fig.update_layout(annotations=[
            dict(
                 x=0.5,
                 y=1.0,
                 showarrow=False,
                 text='Distribution of '+xaxis_column_name+' in '+str(year_value),
                 xref='paper',
                 yref='paper'
                 ),
              ])

    return fig#{
        



def create_time_series(dff, axis_type, title):
    if axis_type=='Nominal':
        y=dff['Nominal ']
    else:
        y=dff['Per_change_previous_year']
    tity=dff.variable.unique()[0]
    return {
        'data': [dict(
            x=dff['year'],
            
            y=y,
            
            mode='lines+markers'
        )],
        'layout': {

            'height': 450,
            'margin': {'l': 100, 'b': 30, 'r': 10, 't': 10},
            'annotations': [{
                'x': 0, 'y': 0.85, 'xanchor': 'left', 'yanchor': 'bottom',
                'xref': 'paper', 'yref': 'paper', 'showarrow': False,
                'align': 'left', 'bgcolor': 'rgba(255, 255, 255, 1)',
                'text': title
            }],
            'yaxis': {'type': 'linear','title':tity },
            'xaxis': {'nticks': 5,

                      'showgrid': False,'title':"Years"}
        }
    }


@app.callback(
    dash.dependencies.Output('x-time-series', 'figure'),
    [dash.dependencies.Input('crossfilter-indicator-scatter', 'hoverData'),
     dash.dependencies.Input('crossfilter-xaxis-column', 'value'),
     dash.dependencies.Input('crossfilter-xaxis-type', 'value')])
def update_y_timeseries(hoverData, xaxis_column_name, axis_type):
    country_name = hoverData['points'][0]['location']
    dff = df[df['id_neighbourhood'] == country_name]
    dff = dff[dff['variable'] == xaxis_column_name]
    title = '<b>{}</b><br>{}'.format(country_name, xaxis_column_name)
    return create_time_series(dff, axis_type, title)


@app.callback(
    dash.dependencies.Output('y-time-series', 'figure'),
    [dash.dependencies.Input('crossfilter-indicator-scatter', 'hoverData')])#,
     
def update_x_timeseries(hoverData):
    
    fig_indexed= go.Figure()
    df_indexed=df[df['id_neighbourhood']==hoverData['points'][0]['location']]
    for i in range(2):
        fig_indexed.add_trace(go.Scatter(x=df_indexed[df_indexed.variable==labels[i]]['year'], y=df_indexed[df_indexed.variable==labels[i]]['Indexed'], mode='lines',
            name=labels[i],
            line=dict(color=colors[i], width=line_size[i]),
            connectgaps=True,
             ))      
    fig_indexed.update_layout(xaxis= {'nticks': 5})
    fig_indexed.update_layout(annotations=[
                            dict(xref='paper', yref='paper', x=0.0, y=1.05,
                              xanchor='left', yanchor='bottom',
                              text='Indexed graph in the period 2015-2019 for '+df_indexed['neighbourhood'].unique()[0],
                              font=dict(family='Arial',
                                        size=16,
                                        color='rgb(37,37,37)'),
                              showarrow=False)
                                ])
    return fig_indexed



@app.callback(
    dash.dependencies.Output('scatter-regression', 'figure'),
    [dash.dependencies.Input('crossfilter-indicator-scatter', 'hoverData')])
def highlight_point(hoverData):
    xy=pd.DataFrame()
    xy['neighbourhood']=df['id_neighbourhood'].unique()
    x=list(df[(df.year==2019)&(df.variable=='Average weight of airbnb inside the neighbourhood')]['Nominal '])-(df[(df.year==2015)&(df.variable=='Average weight of airbnb inside the neighbourhood')]['Nominal '])
    y=(list(df[(df.year==2019)&(df.variable=='Average renting price/m2')]['Nominal '])-df[(df.year==2015)&(df.variable=='Average renting price/m2')]['Nominal '])/df[(df.year==2015)&(df.variable=='Average renting price/m2')]['Nominal ']
    xy['x']=x.values
    xy['y']=y.values
    xy['total_houses']=total_houses['total_houses'].values
    xy['name']=df['neighbourhood'].unique()
    if len(hoverData) == 1:
        point_highlight=(go.Scatter(
                    x=xy[xy['neighbourhood']==hoverData['points'][0]['location']]['x'],
                    y=xy[xy['neighbourhood']==hoverData['points'][0]['location']]['y'],
                    mode='markers',
                    showlegend=False,
                    marker=go.Marker(size=14, line={'width': 5}, color='orange', symbol='circle-open')
                    ))    
        if len(fig_scatter['data']) == 2:
                    fig_scatter['data'][1] = point_highlight
        else:
                    fig_scatter['data'].append(point_highlight)
    return fig_scatter

    
if __name__ == '__main__':
    app.run_server(debug=True)
