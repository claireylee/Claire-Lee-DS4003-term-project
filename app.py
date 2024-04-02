# %%
import dash 
from dash import html, dcc
from dash.dependencies import Input, Output
import plotly.express as px 
import pandas as pd 

# %%
data = pd.read_csv("Song.csv")

# %%
# initialize the dash app 
app = dash.Dash(__name__)
server = app.server


# %%
# app layout 
app.layout = html.Div([
    html.Div([
        html.Button('Sales', id = 'btn-sales', n_clicks = 0), 
        html.Button('Streams', id = 'btn-streams', n_clicks = 0), 
        html.Button('Downloads', id = 'btn-downloads', n_clicks = 0), 
        html.Button('Radio Plays', id = 'btn-radio-plays', n_clicks = 0),
    ]), 
    html.Div(id = 'page-content')
])

# %%
# callback to update the page content based on button clicks
@app.callback(
    Output('page-content', 'children'),
    [Input('btn-sales', 'n_clicks'),
     Input('btn-streams', 'n_clicks'),
     Input('btn-downloads', 'n_clicks'),
     Input('btn-radio-plays', 'n_clicks')]
)
def update_page(btn_sales, btn_streams, btn_downloads, btn_radio_plays):
    ctx = dash.callback_context
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'btn-sales':
        sales_chart = px.bar(data, x = 'Year', y = 'Sales', title = 'Sales Over Years')
        return dcc.Graph(figure = sales_chart)
        return html.Div([
            html.H1('Sales Page'),
        ])
    elif button_id == 'btn-streams':
        # Replace with your streams page content
        return html.Div([
            html.H1('Streams Page'),
            # Add your streams-related content here
        ])
    elif button_id == 'btn-downloads':
        # Replace with your downloads page content
        return html.Div([
            html.H1('Downloads Page'),
            # Add your downloads-related content here
        ])
    elif button_id == 'btn-radio-plays':
        # Replace with your radio plays page content
        return html.Div([
            html.H1('Radio Plays Page'),
            # Add your radio plays-related content here
        ])
    else:
        return html.Div('Welcome! Please select a category.')

# %%
if __name__ == '__main__':
    app.run_server(debug=True)


