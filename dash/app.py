import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import dash_table
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

cities = pd.read_csv('data/cities.csv')

# Initialize app
app = dash.Dash(__name__)

server = app.server

# Layout
app.layout = html.Div([
    # Title Section
    html.H1("Quality of Life City Comparison Application", style={'textAlign': 'center'}),
    
    html.Div([
        # Sidebar Menu
        html.Div([
            # Population Range Slider
            html.Label('Population Filter (Range)'),
            dcc.RangeSlider(
                id='population-slider',
                min=0,
                max=9,
                marks={i: f'{100_000 * i}' if i < 5 else f'{1_000_000 * (i-4)}' for i in range(10)},
                value=[2, 7],
                step=None
            ),
            
            html.Br(),
            # Dropdown for selecting country
            html.Label('Select Country'),
            dcc.Dropdown(
                id='country-dropdown',
                options=[{'label': country, 'value': country} for country in cities['Country Name'].unique()],
                value='All',
                multi=True,
                placeholder="Select Country"
            ),
    
            html.Br(),
            # Dropdown for selecting cities
            html.Label('Select City'),
            dcc.Dropdown(
                id='city-dropdown',
                options=[],
                value=None,
                multi=True,
                placeholder="Select City"
            ),
            
            html.Br(),
            # Dropdown for selecting multiple variables for ranking
            html.Label('Select Variables for Ranking'),
            dcc.Dropdown(
                id='ranking-variable-dropdown',
                options=[
                    {'label': 'Ticket One Way (EUR)', 'value': 'ticket_one_way_eur'},
                    {'label': 'Monthly Pass (EUR)', 'value': 'monthly_pass_eur'},
                    {'label': 'Gasoline per Liter (EUR)', 'value': 'gasoline_liter_eur'},
                    {'label': 'Fitness Club Monthly (EUR)', 'value': 'fitness_club_monthly_eur'},
                    {'label': 'Rent 1BR City Center (EUR)', 'value': 'rent_1br_city_center_eur'},
                    {'label': 'Rent 1BR Outside Center (EUR)', 'value': 'rent_1br_outside_center_eur'},
                    {'label': 'Rent 3BR City Center (EUR)', 'value': 'rent_3br_city_center_eur'},
                    {'label': 'Rent 3BR Outside Center (EUR)', 'value': 'rent_3br_outside_center_eur'},
                    {'label': 'Net Salary Avg (EUR)', 'value': 'net_salary_avg_eur'}
                ],
                value=None,
                multi=True,
                placeholder="Select Variables for Ranking"
            )
        ], style={'padding': 10, 'flex': 1, 'width': '25%', 'display': 'inline-block', 'verticalAlign': 'top'}),
    
        # Map
        html.Div([
            dcc.Graph(id="city-map")
        ], style={'width': '70%', 'display': 'inline-block', 'verticalAlign': 'top'}),
    ], style={'display': 'flex', 'justifyContent': 'center'}),

    # Table for ranking
    html.Div([
        dash_table.DataTable(
            id='ranking-table',
            columns=[
                {'name': 'Rank', 'id': 'Rank'},
                {'name': 'City', 'id': 'City'},
                {'name': 'Country Name', 'id': 'Country Name'},
                {'name': 'Score', 'id': 'Score'}
            ],
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'left'},
            sort_action='native'
        )
    ], style={'width': '90%', 'padding-top': '20px'})
])


# Callback to update the city dropdown based on the selected country
@app.callback(
    Output('city-dropdown', 'options'),
    [Input('country-dropdown', 'value')]
)
def update_city_dropdown(selected_countries):
    if not selected_countries or 'All' in selected_countries:
        return [{'label': city, 'value': city} for city in cities['City']]
    else:
        filtered_cities = cities[cities['Country Name'].isin(selected_countries)]
        return [{'label': city, 'value': city} for city in filtered_cities['City']]


# Callback to update the map based on selected filters
@app.callback(
    Output("city-map", "figure"),
    [Input('population-slider', 'value'),
     Input('country-dropdown', 'value'),
     Input('city-dropdown', 'value')]
)
def update_map(population_value, selected_countries, selected_cities):
    # Convert slider value into population ranges
    population_min = 100_000 * population_value[0] if population_value[0] < 5 else 1_000_000 * (population_value[0] - 4)
    population_max = 100_000 * population_value[1] if population_value[1] < 5 else 1_000_000 * (population_value[1] - 4)
    
    # Filter cities based on population range
    filtered_cities = cities[(cities['Population'] >= population_min) & (cities['Population'] <= population_max)]
    
    # Filter based on selected countries
    if selected_countries and 'All' not in selected_countries:
        filtered_cities = filtered_cities[filtered_cities['Country Name'].isin(selected_countries)]
    
    # Filter based on selected cities
    if selected_cities:
        filtered_cities = filtered_cities[filtered_cities['City'].isin(selected_cities)]
    
    # Create map figure
    fig = px.scatter_mapbox(
        filtered_cities,
        lat="Latitude",
        lon="Longitude",
        hover_name="City",
        zoom=3,
        mapbox_style="open-street-map"
    )
    fig.update_traces(marker=dict(size=8, color='blue'), selector=dict(mode='markers'))
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0}, mapbox_center={"lat": 50, "lon": 10})

    return fig


# Callback to update the ranking table based on selected filters and variables
@app.callback(
    Output('ranking-table', 'data'),
    [Input('population-slider', 'value'),
     Input('country-dropdown', 'value'),
     Input('city-dropdown', 'value'),
     Input('ranking-variable-dropdown', 'value')]
)
def update_ranking_table(population_value, selected_countries, selected_cities, selected_variables):
    # Convert slider value into population ranges
    population_min = 100_000 * population_value[0] if population_value[0] < 5 else 1_000_000 * (population_value[0] - 4)
    population_max = 100_000 * population_value[1] if population_value[1] < 5 else 1_000_000 * (population_value[1] - 4)
    
    # Filter cities based on population range
    filtered_cities = cities[(cities['Population'] >= population_min) & (cities['Population'] <= population_max)]
    
    # Filter based on selected countries
    if selected_countries and 'All' not in selected_countries:
        filtered_cities = filtered_cities[filtered_cities['Country Name'].isin(selected_countries)]
    
    # Filter based on selected cities
    if selected_cities:
        filtered_cities = filtered_cities[filtered_cities['City'].isin(selected_cities)]
    
    # Normalize values and rank cities based on selected variables
    if selected_variables:
        scaler = MinMaxScaler()
        filtered_cities[selected_variables] = scaler.fit_transform(filtered_cities[selected_variables])
        filtered_cities['Score'] = filtered_cities[selected_variables].sum(axis=1)
        ranked_cities = filtered_cities[['City', 'Country Name', 'Score']].sort_values(by='Score', ascending=False)
        ranked_cities['Rank'] = ranked_cities['Score'].rank(method='min', ascending=False).astype(int)
        
        return ranked_cities[['Rank', 'City', 'Country Name', 'Score']].to_dict('records')
    else:
        return []


# Run the app
if __name__ == "__main__":
    app.run_server(debug=True, port=8050)
