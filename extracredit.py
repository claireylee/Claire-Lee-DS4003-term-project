# %%

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, log_loss
from dash import Dash, html, dcc, Input, Output
import plotly.graph_objs as go



# loading the data
data = pd.read_csv("match_data.csv", low_memory=False)

# missing data and define columns 
numeric_cols = ['Field Goals Percentage', 'Free Throws Percentage', 'Three Pointers Percentage', 'Rebounds Defensive', 'Rebounds Offensive', 'Efficiency']
data[numeric_cols] = data[numeric_cols].apply(pd.to_numeric, errors='coerce').fillna(data[numeric_cols].mean())

# winner calculation 
data['Winner'] = data.apply(lambda row: row['Basketball Matches - Match → Home Name'] if row['Basketball Matches - Match → Home Score'] > row['Basketball Matches - Match → Away Score'] else row['Basketball Matches - Match → Away Name'], axis=1)

# win counts
data['Wins'] = data.groupby('Winner')['Match ID'].transform('count')

# handle NaNs
data['Wins'] = data['Wins'].fillna(0) 
data['Wins'] = data['Wins'].astype(int)  

# stages
max_wins = data['Wins'].max()
stage_mapping = {i: "Did Not Qualify" for i in range(max_wins + 1)}
unique_stages = [
    "Did Not Qualify",
    "Round of 64",
    "Round of 32",
    "Sweet Sixteen",
    "Elite Eight",
    "Final Four",
    "Championship Game",
    "Champion"
]
for i in range(1, max_wins + 1):
    if i == 1:
        stage_mapping[i] = "Round of 64"
    elif i == 2:
        stage_mapping[i] = "Round of 32"
    elif i == 3:
        stage_mapping[i] = "Sweet Sixteen"
    elif i == 4:
        stage_mapping[i] = "Elite Eight"
    elif i == 5:
        stage_mapping[i] = "Final Four"
    elif i == 6:
        stage_mapping[i] = "Championship Game"
    elif i >= 7:
        stage_mapping[i] = "Champion"

data['Estimated Stage'] = data['Wins'].apply(lambda x: stage_mapping[x])

# necessary to use stages
data['Stage Code'] = pd.Categorical(data['Estimated Stage'], categories=unique_stages, ordered=True).codes

# modeling setup
X = data[numeric_cols]
y = data['Stage Code']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# calculations
weights = y_train.value_counts()
class_weights = {i: (1 / count) * y_train.size / len(np.unique(y_train)) for i, count in weights.items()}

# SCALING
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('logistic', LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000, class_weight=class_weights))
])
pipeline.fit(X_train, y_train)

# save model
with open('ncaa_tournament_model.pkl', 'wb') as f:
    pickle.dump(pipeline, f)


def get_model_predictions(team_name, data):
    team_data = data[data['Club Name'] == team_name]
    if team_data.empty:
        return {stage: 0 for stage in stage_mapping.values()}
    latest_data = team_data[numeric_cols].iloc[-1:].fillna(method='ffill')
    probabilities = pipeline.predict_proba(latest_data)[0]
    stage_probabilities = dict(zip(stage_mapping.values(), probabilities))
    print("Debug - Stage Probabilities:", stage_probabilities)  # Debugging line to check probabilities
    return stage_probabilities

#app
app = Dash(__name__)


# layout
app.layout = html.Div([
    html.H1("NCAA Basketball Tournament Predictions"),
    dcc.Dropdown(
        id='team-selector',
        options=[{'label': team, 'value': team} for team in data['Club Name'].dropna().unique()],
        value='Select a Team',
        clearable=False
    ),
    dcc.Graph(id='prediction-graph')
])

@app.callback(
    Output('prediction-graph', 'figure'),
    [Input('team-selector', 'value')]
)
def update_graph(selected_team):
    if not selected_team or selected_team == 'Select a Team':
        return go.Figure()
    model_predictions = get_model_predictions(selected_team, data)
    categories = list(stage_mapping.values())  # check for order
    probabilities = [model_predictions.get(stage, 0) for stage in categories]
    fig = go.Figure([go.Bar(x=categories, y=probabilities, name='Probability')])
    fig.update_layout(
        title=f'Probabilities of {selected_team} Advancing to Each Stage',
        xaxis_title='Tournament Stages',
        yaxis_title='Probability',
        yaxis=dict(range=[0, 1])
    )
    return fig

if __name__ == '__main__':
    app.run_server(debug = True)


