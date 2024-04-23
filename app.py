# %%
import dash 
from dash import html, dcc
from dash.dependencies import Input, Output, State
import plotly.express as px 
import pandas as pd 
import plotly.graph_objects as go
import json
import time

# %%
# initialize the dash app 
app = dash.Dash(__name__)
server = app.server

# %%
data = pd.read_csv("data/Song.csv")
data.sort_values('Year', inplace = True)

#grouping data by year
categories = ['Sales', 'Streams', 'Downloads', 'Radio Plays']
yearly_data = {cat: data.groupby('Year')[cat].sum().reset_index() for cat in categories}

#defining positive
positive_words = ['Happy', 'Joy', 'Smile', 'Dream', 'Beautiful', 'Magic', 'Sunshine', 'Heaven', 'Dance', 'Heart', 'Freedom', 'Hope', 'Shine', 'Paradise','Victory', 'Celebrate','Miracle', 'Treasure', 'Angel', 'Glory']
negative_words = ['Sad', 'Plain', 'Tears', 'Lonely','Broken', 'Hate', 'Lies', 'Fear', 'Dark', 'Death', 'Cry', 'Scar', 'War', 'Storm', 'Revenge', 'Chaos', 'Rage', 'Tragedy']

# %%
def generate_table(dataframe, max_rows=4850):
    """Generate a Plotly table with a transparent background behind the chart."""
    return go.Figure(data=[go.Table(
        header=dict(
            values=list(dataframe.columns),
            fill_color='white',  # Color for headers
            align='left',
            font=dict(size=14, color='black')  # Text color in headers
        ),
        cells=dict(
            values=[dataframe[col].head(max_rows) for col in dataframe.columns],
            fill_color='lavender',  # Color for cells
            align='left',
            font=dict(size=12, color='black')  # Text color in cells
        )
    )],
    layout=go.Layout(
        width=800,  # Adjust width 
        height=500,  # Adjust height
        paper_bgcolor='rgba(0,0,0,0)',  # Transparent background behind the chart
        plot_bgcolor='rgba(0,0,0,0)' 
    ))

# %%
app.layout = html.Div([
    html.Div([
        html.H1("Top Songs of the World Dashboard", style={'margin-top': '20px', 'color': 'white'}),
        html.P("Click the buttons below to view different analyses of top songs from 1901-2014", style={'color': 'white'}),
        html.Button("Comparison of Popular Media Metrics", id='btn-overview', n_clicks=0, style={'margin': '10px', 'width': '50%', 'alignItems': 'center'}),
        html.Button("Analysis of Song Titles", id='btn-additional', n_clicks=0, style={'margin': '10px', 'width': '50%', 'alignItems':'center'}),
        html.Div([
            dcc.Graph(figure=generate_table(data), style={'width': '60%', 'height': '400px'}),  # Adjusted width for the graph
            html.Div([
                html.H2("Dataset Overview"),
                html.P("This data collects the top songs of the year with the consideration of the commercial success, digital presence, and overall popularity. This dataset is from Kaggle but the primary sources of the dataset are reputable music industry websites, official charts, and streaming platforms. This dashboard provides informative graphs that show analysis on this dataset. The link to the actual primary source of the data is https://www.kaggle.com/datasets/shiivvvaam/top-songs-of-the-world", style={'padding': '10px', 'background-color': 'black', 'color': 'white', 'border-radius': '5px', 'display': 'flex', 'flexDirection': 'column', 'justifyContent': 'center'})
            ], style={'width': '40%', 'padding': '20px', 'display': 'flex', 'flexDirection': 'column', 'justifyContent': 'center', 'color':'white'})
        ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center', 'height': 'auto', 'margin': '0 auto 20px auto'})
    ], id='welcome-page', style={
        'textAlign': 'center',
        'minHeight': '100vh',
        'display': 'flex',
        'flexDirection': 'column',
        'justifyContent': 'flex-start',  #towards top
        'alignItems': 'center'
    }),
    html.Div([
        html.Button('Back to Main Page', id='back-button', n_clicks=0),
        html.Div(id='graph-content')
    ], id='graph-page', style={'display': 'none'}),
    html.Div([
        html.Button('Back to Main Page', id='back-additional-button', n_clicks=0),
        dcc.Dropdown(
            id='chart-dropdown',
            options=[
                {'label': 'Positive Connotations', 'value': 'pos'},
                {'label': 'Negative Connotations', 'value': 'neg'},
                {'label': 'Mentions of Love', 'value': 'love'}
            ],
            value=None,  # No default selection
            placeholder="Select a chart type",
            style={'width': '50%', 'padding': '4px 20px', 'margin': '20px auto', 'display': 'block', 'lineHeight': '30px', 'border': '1px solid #ccc', 'borderRadius': '5px', 'color': '#333', 'textAlign': 'center'}
        ),
        html.Div("Select an option to see the analysis of words in song titles", id='additional-info-content', style={'textAlign': 'center', 'background-color':'black', 'color':'white','width':'50%','textAlign':'center',  'margin': '20px auto'})
    ], id='additional-page', style={'display': 'none'})
], style={
    'backgroundImage': 'url("/assets/aura background.jpeg")',
    'backgroundSize': 'cover',
    'backgroundPosition': 'center',
    'backgroundRepeat': 'no-repeat',
    'height': '100vh',
    'margin': '0',
    'padding': '0',
    'overflow': 'hidden'
})

# %%
# Callback to manage page visibility
@app.callback(
    [Output('welcome-page', 'style'),
     Output('graph-page', 'style'),
     Output('additional-page', 'style')],
    [Input('btn-overview', 'n_clicks'),
     Input('back-button', 'n_clicks'),
     Input('btn-additional', 'n_clicks'),
     Input('back-additional-button', 'n_clicks')],
    prevent_initial_call=True
)
def update_page_visibility(btn_overview_clicks, back_button_clicks, btn_additional_clicks, back_additional_button_clicks):
    ctx = dash.callback_context
    if not ctx.triggered:
        # If no buttons have been clicked, show only the welcome page
        return {'display': 'block'}, {'display': 'none'}, {'display': 'none'}
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    # Define the default welcome-page style
    welcome_page_style = {
        'display': 'block', 
        'textAlign': 'center', 
        'height': '100vh', 
        'color': 'white'  
    }

    # Update visibility
    if button_id == 'btn-overview':
        return {'display': 'none'}, {'display': 'block'}, {'display': 'none'}
    elif button_id == 'back-button':
        return welcome_page_style, {'display': 'none'}, {'display': 'none'}
    elif button_id == 'btn-additional':
        return {'display': 'none'}, {'display': 'none'}, {'display': 'block'}
    elif button_id == 'back-additional-button':
        return welcome_page_style, {'display': 'none'}, {'display': 'none'}

# %%
def create_combined_graph():
    try:
        fig = go.Figure()
        colors = ['blue', 'green', 'red', 'orange']
        bar_width = 0.30  # Width of each bar

        # Initialize only one trace for each category initially
        for i, category in enumerate(categories):
            category_data = yearly_data[category]
            fig.add_trace(go.Bar(
                x=[category_data['Year'].iloc[0]],
                y=[category_data[category].iloc[0]],
                name=category,
                marker=dict(color=colors[i]),
                width=bar_width
            ))

        fig.update_layout(
            xaxis=dict(
                range=[data['Year'].min(), data['Year'].max()], 
                autorange=False, 
                showgrid=True, 
                gridcolor='white'
            ),
            yaxis=dict(
                range=[0, max(max(yearly_data[cat][cat]) for cat in categories) * 1.1], 
                autorange=False, 
                showgrid=True, 
                gridcolor='white'
            ),
            plot_bgcolor='rgba(0, 0, 0, 1)',  # Transparent background
            paper_bgcolor='rgba(0, 0, 0, 1)',  # Transparent background
            font = dict(color = 'lavender')
        )

        # Creating frames with less frequent updates
        frames = []
        years = sorted(data['Year'].unique())
        step = 5  # Modify step to reduce frame count, adjust based on your data span
        for year in range(0, len(years), step):
            frame_data = []
            for idx, category in enumerate(categories):
                cat_data = yearly_data[category]
                frame_data.append(go.Bar(
                    x=cat_data['Year'].iloc[:year + 1],
                    y=cat_data[category].iloc[:year + 1],
                    marker=dict(color=colors[idx]),
                    width=bar_width
                ))
            frames.append(go.Frame(data=frame_data, name=str(years[year])))

        fig.frames = frames
        fig.update_layout(
            updatemenus=[{
                "type": "buttons",
                "buttons": [{
                    "label": "Play",
                    "method": "animate",
                    "args": [None, {"frame": {"duration": 100, "redraw": True},
                                    "fromcurrent": True, "transition": {"duration": 50}}]
                }]
            }]
        )
        # graph and text box
        return html.Div([
    dcc.Graph(figure=fig),
    html.Div("This graph shows the amount of sales, streams, downloads, and radio plays over the years. Over the years, we can conclude that overall sales are continuously popular. Click the play button to see the animation on how these metrics grew over the years.", style={
        'padding': '10px',
        'marginTop': '20px',
        'width': '70%',
        'font-size': '20px',
        'color': 'black',
        'background-color': 'lavender',  # Keeping this grey or another color for contrast
        'textAlign': 'center',
        'marginLeft': 'auto',
        'marginRight': 'auto'
    })
])
    except Exception as e:
        return html.Div([
            html.H3("Failed to create graph"),
            html.P(str(e))
        ])

# %%
# Callback for displaying the graph
@app.callback(
    Output('graph-content', 'children'),
    [Input('btn-overview', 'n_clicks')],
    prevent_initial_call=True
)
def display_graph(n_clicks):
    if n_clicks and n_clicks > 0:
        return create_combined_graph()
    return None

# %%
@app.callback(
    Output('additional-info-content', 'children'),
    [Input('chart-dropdown', 'value')],
    prevent_initial_call=True
)
def display_additional_info(selected_chart):
    if not selected_chart:
        return "Please select an option from the dropdown to display the chart."

    pos_count = sum(any(word in title for word in positive_words) for title in data['Title'])
    neg_count = sum(any(word in title for word in negative_words) for title in data['Title'])
    love_count = sum('love' in title.lower() for title in data['Title'])  # case-insensitive search
    total_songs = len(data)

    layout = {
        'title_font_color': 'black',
        'plot_bgcolor': 'rgba(0,0,0,0)',
        'paper_bgcolor': 'rgba(0,0,0,0)',
        'margin': dict(t=20, l=0, r=0, b=20),
        'font': dict(family="Helvetica, Arial, sans-serif", size=14, color='black'),
        'autosize': True,
        'width': 400,
        'height': 400,
        'legend': {'bgcolor': 'black', 'font': dict(color='white')}
    }

    fig = go.Figure()

    colors = {
        'pos': ['yellow', 'grey'],  # Positive: Bright green, Other: Dark grey
        'neg': ['red', 'grey'],  # Negative: Bright red, Other: Dark grey
        'love': ['pink', 'grey']  # Love: Purple, Other: Dark grey
    }

    if selected_chart == 'pos':
        fig.add_trace(go.Pie(labels=['Positive', 'Other'], values=[pos_count, total_songs - pos_count],
                             hole=.4, marker_colors=colors['pos'], textinfo='label+percent',
                             hoverinfo='label+percent+value'))
        description = "This chart shows the percentage of song titles with positively connotated words."
    elif selected_chart == 'neg':
        fig.add_trace(go.Pie(labels=['Negative', 'Other'], values=[neg_count, total_songs - neg_count],
                             hole=.4, marker_colors=colors['neg'], textinfo='label+percent',
                             hoverinfo='label+percent+value'))
        description = "This chart shows the percentage of song titles with negatively connotated words."
    elif selected_chart == 'love':
        fig.add_trace(go.Pie(labels=['Love', 'Other'], values=[love_count, total_songs - love_count],
                             hole=.4, marker_colors=colors['love'], textinfo='label+percent',
                             hoverinfo='label+percent+value'))
        description = "This chart shows the percentage of song titles that mention 'love'."

    fig.update_layout(**layout)
    fig.update_traces(textposition='inside', textfont_size=16)

    return html.Div([
        dcc.Graph(figure=fig),
        html.P(description, style={
            'background-color': 'black',
            'color': 'white',
            'padding': '10px',
            'border-radius': '5px',
            'width': '80%',  
            'margin': '20px auto',  
            'text-align': 'center'
        })
    ], style={'display': 'flex', 'flex-direction': 'column', 'align-items': 'center', 'justify-content': 'center', 'height': '100%'})

# %%
if __name__ == '__main__':
    app.run_server(debug=True)


