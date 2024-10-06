import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State, ALL
from dash import dash_table
import plotly.express as px
import json
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import numpy as np


cities = pd.read_csv('data/cities.csv')
with open("data/interest_dict_complete_selection.json", "r") as f:
    interest_categories = json.load(f)

# List of columns for which NaN values should be filled with country-wise averages
columns_to_fill = [
    'public_transport_aggregated', 'education_facilities_aggregated', 'healthcare_services_aggregated',
    'green_spaces_aggregated', 'sports_facilities_aggregated', 'cultural_facilities_aggregated',
    'job_availability_aggregated', 'foreigner_integration_aggregated', 'housing_affordability_aggregated',
    'admin_efficiency_aggregated', 'transport_affordability_aggregated', 'transport_accessibility_aggregated',
    'transport_frequency_aggregated', 'transport_reliability_aggregated', 'ticket_one_way_eur',
    'monthly_pass_eur', 'gasoline_liter_eur', 'fitness_club_monthly_eur', 'rent_1br_city_center_eur',
    'rent_1br_outside_center_eur', 'rent_3br_city_center_eur', 'rent_3br_outside_center_eur', 'net_salary_avg_eur'
]
# Fill NaN values in each column with the country-wise average
for col in columns_to_fill:
    cities[col] = cities.groupby('Country Name')[col].transform(lambda x: x.fillna(x.mean()))


# Normalize fb interest categories to be represented by percentage of how many people in the corresponding city are interested in the interest category
columns_to_normalize = cities.columns[33:336]
cities[columns_to_normalize] = cities[columns_to_normalize].apply(pd.to_numeric, errors='coerce')
cities[columns_to_normalize] = cities[columns_to_normalize].div(cities["total_fb_users"], axis=0)

cities = cities.drop(index=20).reset_index(drop=True)

# Define helper functions
def flatten_categories(data, level=0):
    flattened = []
    for key, value in data.items():
        if isinstance(value, dict) and 'id' in value:
            # If value is a dictionary with sub-categories, recurse
            flattened.append((key, level))  # Add the current category with its level
            sub_categories = {k: v for k, v in value.items() if k != 'id'}
            flattened.extend(flatten_categories(sub_categories, level + 1))  # Add sub-categories with increased level
        else:
            flattened.append((key, level))  # Base case for leaf nodes
    return flattened

# Preprocess the interest categories for the dropdown
flattened_categories = flatten_categories(interest_categories)

# Build the hierarchy from the flattened list
def build_hierarchy(flattened_categories):
    root = []
    stack = []
    for name, level in flattened_categories:
        node = {'label': name, 'children': [], 'level': level}
        # Adjust the stack based on the current level
        while len(stack) > level:
            stack.pop()
        if len(stack) == 0:
            root.append(node)
        else:
            parent = stack[-1]
            parent['children'].append(node)
        stack.append(node)
    return root

# Adjusted create_node_layout function
def create_node_layout(node, prefix=''):
    label = node['label']
    children = node['children']
    level = node['level']
    checkbox_id = {'type': 'fb-interest-checkbox', 'id': label}
    checkbox = dbc.Checkbox(
        id=checkbox_id,
        label=label,
        value=False,
        style={'margin-left': f'{20 * level}px'}
    )
    if children:
        # Create collapsible section using html.Details and html.Summary
        return html.Div([
            html.Details([
                html.Summary(label, style={'fontWeight': 'bold', 'margin-left': f'{20 * level}px'}),
                html.Div(checkbox),
                html.Div(
                    [create_node_layout(child) for child in children],
                    style={'marginLeft': '20px'}
                )
            ])
        ])
    else:
        # Return the checkbox for leaf nodes
        return checkbox

# Create the Accordion with recursive items
def create_accordion(items):
    return html.Div(
        [create_node_layout(item) for item in items]
    )

# Build the hierarchy
hierarchy = build_hierarchy(flattened_categories)







external_stylesheets = [
    dbc.themes.BOOTSTRAP,
    "https://use.fontawesome.com/releases/v5.8.1/css/all.css"  # Font Awesome CSS
]


app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server


label_mapping = {
    'ticket_one_way_eur': 'Public Transport One Way Ticket Price',
    'monthly_pass_eur': 'Public Transport Monthly Pass',
    'gasoline_liter_eur': 'Gasoline Price per Liter',
    'fitness_club_monthly_eur': 'Fitness Club Fee',
    'rent_1br_city_center_eur': 'Rent 1 Bedroom City Center',
    'rent_1br_outside_center_eur': 'Rent 1 Bedromm Outside Center',
    'rent_3br_city_center_eur': 'Rent 3 Bedrooms City Center',
    'rent_3br_outside_center_eur': 'Rent 3 Bedroom Outside Center',
    'net_salary_avg_eur': 'Average Net Salary'
}

cost_vars = ['ticket_one_way_eur', 'monthly_pass_eur', 'gasoline_liter_eur', 'fitness_club_monthly_eur', 
             'rent_1br_city_center_eur', 'rent_1br_outside_center_eur', 'rent_3br_city_center_eur', 
             'rent_3br_outside_center_eur']

salary_var = 'net_salary_avg_eur'

# Satisfaction variable label mapping
satisfaction_vars = {
    'public_transport_aggregated': 'Public Transport',
    'education_facilities_aggregated': 'Education Facilities',
    'healthcare_services_aggregated': 'Healthcare Services',
    'green_spaces_aggregated': 'Green Spaces',
    'sports_facilities_aggregated': 'Sports Facilities',
    'cultural_facilities_aggregated': 'Cultural Facilities',
    'job_availability_aggregated': 'Job Availability',
    'foreigner_integration_aggregated': 'Foreigner Integration',
    'housing_affordability_aggregated': 'Housing Affordability',
    'admin_efficiency_aggregated': 'Admin Efficiency',
    'transport_affordability_aggregated': 'Transport Affordability',
    'transport_accessibility_aggregated': 'Transport Accessibility',
    'transport_frequency_aggregated': 'Transport Frequency',
    'transport_reliability_aggregated': 'Transport Reliability'
}
label_mapping.update(satisfaction_vars)  # Merge satisfaction variable labels with existing labels








# Layout
app.layout = html.Div([
    # Title Section with Background Color
    html.Div([
        html.H1(
            "Personalized European City Comparison Tool",
            style={'textAlign': 'center', 'fontSize': '36px', 'margin': '10px 0'}
        ),
        html.H3(
            "Find your ideal European city based on what matters most to you.",
            style={'textAlign': 'center', 'fontStyle': 'italic', 'margin': '0 0 10px 0'}
        ),
        html.P(
            "Use this application to explore and compare European cities according to your preferences. Adjust the filters and weights to prioritize factors like cost of living, income, satisfaction with urban services, and social interests. Data sources include Numbeo, Eurostat, and Facebook Marketing API.",
            style={'textAlign': 'center', 'fontSize': '14px', 'margin': '0 20px'}
        )
    ], style={
        'backgroundColor': '#e6f7ff',
        'padding': '15px 0',
        'marginLeft': '30%',  # Offset to prevent overlap with the sidebar
    }),

    # Main Content: Sidebar and Main Panel
    html.Div([
        # Sidebar Menu on the Left
        html.Div([
            html.Br(),
            html.H2("Customize Your Preferences", style={"textAlign": "center", 'fontSize': '24px'}),

            html.Br(),

            # Population Filter
            html.Label([
                'Filter Cities by Population Range',
                html.I(className="fas fa-info-circle", id='population-tooltip', style={'margin-left': '5px', 'cursor': 'pointer'})
            ], style={'fontSize': '16px', 'fontWeight': 'bold'}),
            dbc.Tooltip(
                "Use the slider to select cities with populations within your desired range.",
                target='population-tooltip',
                placement='right',
            ),
            dcc.RangeSlider(
                id='population-slider',
                min=0,
                max=11,
                marks={i: f'{i * 100}k' if i <= 5 else (f'{(i - 5) * 1}M' if i < 11 else '20M+') for i in range(12)},
                value=[0, 11],
                step=None
            ),

            html.Br(),

            # Country Filter
            html.Label([
                'Filter by Country',
                html.I(className="fas fa-info-circle", id='country-tooltip', style={'margin-left': '5px', 'cursor': 'pointer'})
            ], style={'fontSize': '16px', 'fontWeight': 'bold'}),
            dbc.Tooltip(
                "Select one or more countries to include cities only from those countries.",
                target='country-tooltip',
                placement='right',
            ),
            dcc.Dropdown(
                id='country-dropdown',
                options=[{'label': country, 'value': country} for country in cities['Country Name'].unique()],
                value='All',
                multi=True,
                placeholder="Select Country"
            ),

            html.Br(),

            # City Filter
            html.Label([
                'Filter by City',
                html.I(className="fas fa-info-circle", id='city-tooltip', style={'margin-left': '5px', 'cursor': 'pointer'})
            ], style={'fontSize': '16px', 'fontWeight': 'bold'}),
            dbc.Tooltip(
                "Select specific cities to focus your comparison.",
                target='city-tooltip',
                placement='right',
            ),
            dcc.Dropdown(
                id='city-dropdown',
                options=[],
                value=None,
                multi=True,
                placeholder="Select City"
            ),

            html.Br(),

            # Income Variable
            html.Label([
                'Income Variable',
                html.I(className="fas fa-info-circle", id='income-tooltip', style={'margin-left': '5px', 'cursor': 'pointer'})
            ], style={'fontSize': '16px', 'fontWeight': 'bold'}),
            dbc.Tooltip(
                "Include average net salary in the comparison. Adjust its importance with the slider.",
                target='income-tooltip',
                placement='right',
            ),
            dcc.Dropdown(
                id='income-dropdown',
                options=[{'label': label_mapping['net_salary_avg_eur'], 'value': 'net_salary_avg_eur'}],
                value=None,  # Do not select by default
                placeholder="Select Income Variable"
            ),
            dcc.Slider(
                id='income-slider',
                min=1, max=5, step=1, value=1,
                marks={i: str(i) for i in range(1, 6)}
            ),

            html.Br(),

            # Cost of Living Variables
            html.Div([
                html.Label([
                    "Cost of Living Variables",
                    html.I(className="fas fa-info-circle", id='cost-tooltip', style={'margin-left': '5px', 'cursor': 'pointer'})
                ], style={'fontSize': '16px', 'fontWeight': 'bold'}),
                dbc.Tooltip(
                    "Select cost of living factors to include in the comparison. Click 'Add' to select more variables.",
                    target='cost-tooltip',
                    placement='right',
                ),
                html.Button('Add Variable', id='add-button', n_clicks=0,
                            style={'margin-left': 'auto', 'display': 'block'})  # Push the button to the right
            ]),
            html.Div(id='cost-variable-dropdowns', children=[]),

            html.Br(),

            # Satisfaction Variables
            html.Div([
                html.Label([
                    "Urban Service Quality",
                    html.I(className="fas fa-info-circle", id='satisfaction-tooltip', style={'margin-left': '5px', 'cursor': 'pointer'})
                ], style={'fontSize': '16px', 'fontWeight': 'bold'}),
                dbc.Tooltip(
                    "Eurostat satisfaction survey data on urban service quality. Click 'Add' to select more variables and adjust its importance with the slider.",
                    target='satisfaction-tooltip',
                    placement='right',
                ),
                html.Button('Add Variable', id='add-satisfaction-button', n_clicks=0,
                            style={'margin-left': 'auto', 'display': 'block'})  # Push the button to the right
            ]),
            html.Div(id='satisfaction-variable-dropdowns', children=[]),

            html.Br(),

            # FB Interest Categories
            html.Label([
                "Select your Personal Interests",
                html.I(className="fas fa-info-circle", id='fb-tooltip', style={'margin-left': '5px', 'cursor': 'pointer'})
            ], style={'fontSize': '16px', 'fontWeight': 'bold'}),
            dbc.Tooltip(
                "Select categories you are interested in to find those cities with the highest amount of people interested in the same categories on Meta services.",
                target='fb-tooltip',
                placement='right',
            ),
            create_accordion(hierarchy),
            html.Br(),
            html.Label([
                "Weight for Social Fit",
                html.I(className="fas fa-info-circle", id='social-fit-tooltip', style={'margin-left': '5px', 'cursor': 'pointer'})
            ], style={'fontSize': '16px', 'fontWeight': 'bold'}),
            dbc.Tooltip(
                "Adjust the importance of social fit in the overall score.",
                target='social-fit-tooltip',
                placement='right',
            ),
            dcc.Slider(
                id='social-fit-slider',
                min=1, max=5, step=1, value=1,
                marks={i: str(i) for i in range(1, 6)}
            ),

        ], style={
            'padding': '20px',
            'width': '30%',
            'backgroundColor': '#f0f0f0',  # Light grey background for the menu
            'boxSizing': 'border-box',
            'overflowY': 'auto',
            'height': '100vh',
            'position': 'fixed',
            'left': '0',
            'top': '0',
            'bottom': '0',
            'zIndex': '1'
        }),

        # Main Panel on the Right
        html.Div([
            # Map Section
            html.Div([
                html.H4([
                    "Map of Cities",
                    html.I(className="fas fa-info-circle", id='map-tooltip', style={'margin-left': '5px', 'cursor': 'pointer'})
                ], style={'fontSize': '20px', 'fontWeight': 'bold'}),
                dbc.Tooltip(
                    "This map shows the cities based on your preferences. Hover over markers for more info.",
                    target='map-tooltip',
                    placement='right',
                ),
                dcc.Graph(id="city-map", style={'height': '600px'}),
            ], style={'padding': '10px'}),

            # Ranking Table
            html.Div([
                html.H4([
                    "City Rankings",
                    html.I(className="fas fa-info-circle", id='ranking-tooltip', style={'margin-left': '5px', 'cursor': 'pointer'})
                ], style={'fontSize': '20px', 'fontWeight': 'bold'}),
                dbc.Tooltip(
                    "This table lists the cities ranked based on your selected preferences and weights.",
                    target='ranking-tooltip',
                    placement='right',
                ),
                dash_table.DataTable(
                    id='ranking-table',
                    columns=[
                        {'name': 'Rank', 'id': 'Rank'},
                        {'name': 'City', 'id': 'City'},
                        {'name': 'Country Name', 'id': 'Country Name'},
                        {'name': 'Score', 'id': 'Score'}
                    ],
                    style_table={
                        'overflowX': 'auto',
                        'width': '100%',
                        'backgroundColor': '#e6f7ff'
                    },
                    style_cell={
                        'textAlign': 'left',
                        'backgroundColor': '#e6f7ff'
                    },
                    sort_action='native'
                )
            ], style={'padding': '10px'}),
        ], style={
            'marginLeft': '30%',  # Offset to match the sidebar
            'width': '70%',       # Occupy the remaining width
            'padding': '20px',
            'backgroundColor': '#e6f7ff',  # Light blue background for the main content
            'boxSizing': 'border-box'
        }),
    ], style={'display': 'flex', 'flexDirection': 'row'}),
])

# Callbacks (same as your original code, adjusted if necessary)

# Callback to update the city dropdown based on the selected country and population range
@app.callback(
    Output('city-dropdown', 'options'),
    [Input('country-dropdown', 'value'), Input('population-slider', 'value')]
)
def update_city_dropdown(selected_countries, population_value):
    # Convert slider value into population ranges
    population_min = 100_000 * population_value[0] if population_value[0] <= 5 else 1_000_000 * (population_value[0] - 5)
    population_max = 100_000 * population_value[1] if population_value[1] <= 5 else 1_000_000 * (population_value[1] - 5)

    # Filter cities based on selected countries and population range
    filtered_cities = cities[(cities['Population'] >= population_min) & (cities['Population'] <= population_max)]

    if selected_countries and 'All' not in selected_countries:
        filtered_cities = filtered_cities[filtered_cities['Country Name'].isin(selected_countries)]

    return [{'label': city, 'value': city} for city in filtered_cities['City']]

# Callback to dynamically add cost variable dropdowns and sliders
@app.callback(
    Output('cost-variable-dropdowns', 'children'),
    Input('add-button', 'n_clicks'),
    State('cost-variable-dropdowns', 'children'),
    State({'type': 'dynamic-dropdown', 'index': ALL}, 'value')
)
def add_cost_dropdown(n_clicks, children, selected_vars):
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate

    # Filter out already selected variables
    selected_vars = selected_vars or []
    available_vars = [{'label': label_mapping[var], 'value': var} for var in cost_vars if var not in selected_vars]

    if not available_vars:  # If no more variables are available, don't add new dropdowns
        return children

    # Define the new dropdown and slider for the new cost variable
    new_element = html.Div([
        dcc.Dropdown(
            id={'type': 'dynamic-dropdown', 'index': n_clicks},
            options=available_vars,  # Filtered options for cost variables
            value=None  # No default selected
        ),
        dcc.Slider(
            id={'type': 'dynamic-slider', 'index': n_clicks},  # A corresponding slider
            min=1, max=5, step=1, value=1,
            marks={i: str(i) for i in range(1, 6)}  # Slider value range from 1 to 5
        ),
        html.Br()
    ])
    # Add the new dropdown and slider to the list of children
    children.append(new_element)
    return children

# Callback to dynamically add satisfaction variable dropdowns and sliders
@app.callback(
    Output('satisfaction-variable-dropdowns', 'children'),
    Input('add-satisfaction-button', 'n_clicks'),
    State('satisfaction-variable-dropdowns', 'children'),
    State({'type': 'dynamic-satisfaction-dropdown', 'index': ALL}, 'value')
)
def add_satisfaction_dropdown(n_clicks, children, selected_vars):
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate

    # Filter out already selected variables
    selected_vars = selected_vars or []
    available_vars = [{'label': label_mapping[var], 'value': var} for var in satisfaction_vars if var not in selected_vars]

    if not available_vars:  # If no more variables are available, don't add new dropdowns
        return children

    # Define the new dropdown and slider for the new satisfaction variable
    new_element = html.Div([
        dcc.Dropdown(
            id={'type': 'dynamic-satisfaction-dropdown', 'index': n_clicks},
            options=available_vars,  # Filtered options for satisfaction variables
            value=None  # No default selected
        ),
        dcc.Slider(
            id={'type': 'dynamic-satisfaction-slider', 'index': n_clicks},  # A corresponding slider
            min=1, max=5, step=1, value=1,
            marks={i: str(i) for i in range(1, 6)}  # Slider value range from 1 to 5
        ),
        html.Br()
    ])
    # Add the new dropdown and slider to the list of children
    children.append(new_element)
    return children




@app.callback(
    Output("city-map", "figure"),
    [
        Input('population-slider', 'value'),
        Input('country-dropdown', 'value'),
        Input('city-dropdown', 'value'),
        Input({'type': 'dynamic-dropdown', 'index': ALL}, 'value'),
        Input({'type': 'dynamic-slider', 'index': ALL}, 'value'),
        Input('income-dropdown', 'value'),
        Input('income-slider', 'value'),
        Input({'type': 'dynamic-satisfaction-dropdown', 'index': ALL}, 'value'),
        Input({'type': 'dynamic-satisfaction-slider', 'index': ALL}, 'value'),
        Input({'type': 'fb-interest-checkbox', 'id': ALL}, 'value'),
        State({'type': 'fb-interest-checkbox', 'id': ALL}, 'id'),
        Input('social-fit-slider', 'value')
    ]
)
def update_map(population_value, selected_countries, selected_cities,
               selected_variables, variable_weights,
               income_variable, income_weight,
               selected_satisfaction_vars, satisfaction_weights,
               fb_interest_values, fb_interest_ids,
               social_fit_weight):

    # Convert slider value into population ranges
    population_min = 100_000 * population_value[0] if population_value[0] <= 5 else 1_000_000 * (population_value[0] - 5)
    population_max = 100_000 * population_value[1] if population_value[1] <= 5 else 1_000_000 * (population_value[1] - 5)

    if population_value[1] == 12:
        population_max = 20_000_000

    # Filter cities based on population range
    filtered_cities = cities[(cities['Population'] >= population_min) & (cities['Population'] <= population_max)].copy()

    # Filter based on selected countries
    if selected_countries and 'All' not in selected_countries:
        filtered_cities = filtered_cities[filtered_cities['Country Name'].isin(selected_countries)]

    # Filter based on selected cities
    if selected_cities:
        filtered_cities = filtered_cities[filtered_cities['City'].isin(selected_cities)]

    if filtered_cities.empty:
        return go.Figure()  # Return empty figure if no cities

    # Handle selected cost of living and satisfaction variables
    selected_variables = [var for var in selected_variables if var is not None]
    selected_satisfaction_vars = [var for var in selected_satisfaction_vars if var is not None]
    variable_weights = variable_weights[:len(selected_variables)]
    satisfaction_weights = satisfaction_weights[:len(selected_satisfaction_vars)]

    # Add income variable if selected
    if income_variable is not None:
        selected_variables.append(income_variable)
        variable_weights.append(income_weight)

    # Prepare normalization for variables (min-max scaling)
    scaler = MinMaxScaler()

    weighted_scores = []

    # Handle cost of living variables
    reverse_vars = [
        'ticket_one_way_eur', 'monthly_pass_eur', 'gasoline_liter_eur',
        'fitness_club_monthly_eur', 'rent_1br_city_center_eur',
        'rent_1br_outside_center_eur', 'rent_3br_city_center_eur',
        'rent_3br_outside_center_eur'
    ]

    # Normalize and weight cost of living variables
    for idx, var in enumerate(selected_variables):
        if var in filtered_cities.columns:
            data = filtered_cities[[var]].astype(float)
            if var in reverse_vars:
                normalized = 1 - scaler.fit_transform(data)
            else:
                normalized = scaler.fit_transform(data)
            weight = variable_weights[idx] if idx < len(variable_weights) else 1
            weighted_scores.append(normalized * weight)
        else:
            # If variable not in columns, append zeros
            weighted_scores.append(np.zeros((len(filtered_cities), 1)))

    # Normalize and weight satisfaction variables
    for idx, var in enumerate(selected_satisfaction_vars):
        if var in filtered_cities.columns:
            data = filtered_cities[[var]].astype(float)
            normalized = scaler.fit_transform(data)
            weight = satisfaction_weights[idx] if idx < len(satisfaction_weights) else 1
            weighted_scores.append(normalized * weight)
        else:
            # If variable not in columns, append zeros
            weighted_scores.append(np.zeros((len(filtered_cities), 1)))

    # Process FB interest categories
    selected_fb_interests = [fb_interest_ids[i]['id'] for i, val in enumerate(fb_interest_values) if val]
    if selected_fb_interests:
        fb_interest_columns = [col for col in selected_fb_interests if col in filtered_cities.columns]
        if fb_interest_columns:
            # For scoring, sum the selected columns
            filtered_cities['Social Fit'] = filtered_cities[fb_interest_columns].sum(axis=1)
            # Normalize the 'Social Fit' score from 0 to 1
            filtered_cities['Social Fit'] = scaler.fit_transform(filtered_cities[['Social Fit']])
            # Multiply by the social fit weight
            weighted_scores.append(filtered_cities[['Social Fit']] * social_fit_weight)
        else:
            filtered_cities['Social Fit'] = 0
    else:
        filtered_cities['Social Fit'] = 0

    # Calculate final score only if there are any selected variables
    if weighted_scores:
        total_weighted_scores = np.sum(weighted_scores, axis=0)
        # Sum across all variables to get a single score per city
        filtered_cities['Score'] = total_weighted_scores.sum(axis=1)
        filtered_cities['Score'] = filtered_cities['Score'].round(3)
    else:
        filtered_cities['Score'] = 0  # Set score to 0 if no variables are selected

    # Handle case where all scores are zero
    if (filtered_cities['Score'] > 0).any():
        min_score = filtered_cities['Score'][filtered_cities['Score'] > 0].min()
        max_score = filtered_cities['Score'].max()
    else:
        min_score = 0
        max_score = 1  # Avoid division by zero

    # Create the background trace for the outline
    background_trace = go.Scattermapbox(
        lat=filtered_cities['Latitude'],
        lon=filtered_cities['Longitude'],
        mode='markers',
        marker=dict(
            size=13,  # Slightly larger size for the outline
            color='black',
            opacity=1
        ),
        hoverinfo='none',  # We don't need hover info for the outline
        showlegend=False
    )

    # Create the foreground trace for the actual markers
    foreground_trace = go.Scattermapbox(
        lat=filtered_cities['Latitude'],
        lon=filtered_cities['Longitude'],
        mode='markers',
        marker=dict(
            size=11,
            color=filtered_cities['Score'],
            colorscale='RdYlGn',
            cmin=min_score,
            cmax=max_score,
            colorbar=dict(
                title='Score',
                x=0.95,  # Adjust to position the color scale inside the map
                xanchor='left',
                y=0.5
            )
        ),
        hovertext=filtered_cities['City'],
        customdata=filtered_cities[['Country Name', 'Population', 'Score']],
        hovertemplate=(
            "<b>%{hovertext}</b><br>" +
            "Country Name: %{customdata[0]}<br>" +
            "Population: %{customdata[1]}<br>" +
            "Score: %{customdata[2]}<extra></extra>"
        ),
        showlegend=False
    )

    # Create the figure with both traces
    fig = go.Figure(data=[background_trace, foreground_trace])

    # Update the layout
    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            zoom=3.6,
            center={"lat": 50, "lon": 10}
        ),
        margin={"r": 0, "t": 0, "l": 0, "b": 0}
    )

    return fig





@app.callback(
    Output('ranking-table', 'columns'),
    [
        Input({'type': 'dynamic-dropdown', 'index': ALL}, 'value'),
        Input({'type': 'dynamic-satisfaction-dropdown', 'index': ALL}, 'value'),
        Input('income-dropdown', 'value'),
        Input({'type': 'fb-interest-checkbox', 'id': ALL}, 'value'),
        State({'type': 'fb-interest-checkbox', 'id': ALL}, 'id')
    ]
)
def update_table_columns(selected_variables, selected_satisfaction_vars, income_variable,
                         fb_interest_values, fb_interest_ids):
    base_columns = [
        {'name': 'Rank', 'id': 'Rank'},
        {'name': 'City', 'id': 'City'},
        {'name': 'Country Name', 'id': 'Country Name'},
        {'name': 'Score', 'id': 'Score'}
    ]
    
    # Include the income variable in the list of columns
    all_selected_vars = [var for var in selected_variables if var] + [var for var in selected_satisfaction_vars if var]
    if income_variable is not None:
        all_selected_vars.append(income_variable)
    
    variable_columns = [{'name': label_mapping.get(var, var), 'id': var} for var in all_selected_vars]
    
    # Determine if any FB interest categories are selected
    selected_fb_interests = [fb_interest_ids[i]['id'] for i, val in enumerate(fb_interest_values) if val]
    if selected_fb_interests:
        variable_columns.append({'name': 'Social Fit', 'id': 'Social Fit Display'})

    
    return base_columns + variable_columns






@app.callback(
    Output('ranking-table', 'data'),
    [
        Input('population-slider', 'value'),
        Input('country-dropdown', 'value'),
        Input('city-dropdown', 'value'),
        Input({'type': 'dynamic-dropdown', 'index': ALL}, 'value'),
        Input({'type': 'dynamic-slider', 'index': ALL}, 'value'),
        Input('income-dropdown', 'value'),
        Input('income-slider', 'value'),
        Input({'type': 'dynamic-satisfaction-dropdown', 'index': ALL}, 'value'),
        Input({'type': 'dynamic-satisfaction-slider', 'index': ALL}, 'value'),
        Input({'type': 'fb-interest-checkbox', 'id': ALL}, 'value'),
        State({'type': 'fb-interest-checkbox', 'id': ALL}, 'id'),
        Input('social-fit-slider', 'value')
    ]
)
def update_ranking_table(population_value, selected_countries, selected_cities,
                         selected_variables, variable_weights,
                         income_variable, income_weight,
                         selected_satisfaction_vars, satisfaction_weights,
                         fb_interest_values, fb_interest_ids,
                         social_fit_weight):

    
    # Convert slider value into population ranges
    population_min = 100_000 * population_value[0] if population_value[0] <= 5 else 1_000_000 * (population_value[0] - 5)
    population_max = 100_000 * population_value[1] if population_value[1] <= 5 else 1_000_000 * (population_value[1] - 5)

    # Filter cities based on population range
    filtered_cities = cities[(cities['Population'] >= population_min) & (cities['Population'] <= population_max)]
    
    # Filter based on selected countries
    if selected_countries and 'All' not in selected_countries:
        filtered_cities = filtered_cities[filtered_cities['Country Name'].isin(selected_countries)]

    # Filter based on selected cities
    if selected_cities:
        filtered_cities = filtered_cities[filtered_cities['City'].isin(selected_cities)]

    if filtered_cities.empty:
        return []  # Prevents crash if there are no cities

    # Handle selected cost of living and satisfaction variables
    selected_variables = [var for var in selected_variables if var is not None]
    selected_satisfaction_vars = [var for var in selected_satisfaction_vars if var is not None]
    variable_weights = variable_weights[:len(selected_variables)]
    satisfaction_weights = satisfaction_weights[:len(selected_satisfaction_vars)]

    # Add income variable if selected
    if income_variable is not None:
        selected_variables.append(income_variable)
        variable_weights.append(income_weight)

    # Prepare normalization for variables (min-max scaling)
    scaler = MinMaxScaler()

    weighted_scores = []
    
    # Handle cost of living variables
    reverse_vars = [
        'ticket_one_way_eur', 'monthly_pass_eur', 'gasoline_liter_eur',
        'fitness_club_monthly_eur', 'rent_1br_city_center_eur',
        'rent_1br_outside_center_eur', 'rent_3br_city_center_eur', 
        'rent_3br_outside_center_eur'
    ]

    # Normalize cost of living variables
    for idx, var in enumerate(selected_variables):
        if var in filtered_cities.columns:
            if var in reverse_vars:
                normalized = 1 - scaler.fit_transform(filtered_cities[[var]])
            else:
                normalized = scaler.fit_transform(filtered_cities[[var]])
            weight = variable_weights[idx] if idx < len(variable_weights) else 1
            weighted_scores.append(normalized * weight)

    # Normalize satisfaction variables
    for idx, var in enumerate(selected_satisfaction_vars):
        if var in filtered_cities.columns:
            normalized = scaler.fit_transform(filtered_cities[[var]])
            satisfaction_weight = satisfaction_weights[idx] if idx < len(satisfaction_weights) else 1
            weighted_scores.append(normalized * satisfaction_weight)

    # Process FB interest categories
    # Process FB interest categories
    selected_fb_interests = [fb_interest_ids[i]['id'] for i, val in enumerate(fb_interest_values) if val]
    if selected_fb_interests:
        fb_interest_columns = [col for col in selected_fb_interests if col in filtered_cities.columns]
        if fb_interest_columns:
            # Calculate average of selected categories for each city
            filtered_cities['Social Fit Display'] = filtered_cities[fb_interest_columns].mean(axis=1) * 100
            filtered_cities['Social Fit Display'] = filtered_cities['Social Fit Display'].round(2).astype(str) + '%'
    
            # For scoring, sum the selected columns
            filtered_cities['Social Fit'] = filtered_cities[fb_interest_columns].sum(axis=1)
            # Normalize the 'Social Fit' score from 0 to 1
            filtered_cities['Social Fit'] = MinMaxScaler().fit_transform(filtered_cities[['Social Fit']])
            # Multiply by the social fit weight
            weighted_scores.append(filtered_cities[['Social Fit']] * social_fit_weight)
        else:
            # No matching columns found
            filtered_cities['Social Fit'] = 0
            filtered_cities['Social Fit Display'] = '0%'
    else:
        filtered_cities['Social Fit'] = 0
        filtered_cities['Social Fit Display'] = '0%'


    # Calculate final score only if there are any selected variables
    if weighted_scores:
        total_weighted_scores = sum(weighted_scores)
        filtered_cities['Score'] = total_weighted_scores.round(3)
    else:
        filtered_cities['Score'] = 0  # Set score to 0 if no variables are selected

    # Combine original values with scores and rank
    columns_to_include = selected_variables + selected_satisfaction_vars
    if 'Social Fit Display' in filtered_cities.columns:
        columns_to_include.append('Social Fit Display')


    ranked_cities = filtered_cities[['City', 'Country Name', 'Score']].join(filtered_cities[columns_to_include])
    
    ranked_cities['Rank'] = ranked_cities['Score'].rank(method='min', ascending=False)
    ranked_cities = ranked_cities.sort_values(by='Rank', ascending=True)
    
    # Add "€" to cost-related variables for display
    for var in selected_variables:
        if var in reverse_vars:  # Only for cost-related variables
            ranked_cities[var] = ranked_cities[var].apply(lambda x: f"{x:.2f}€")

    # Add "€" to net salary variable for display
    if income_variable == 'net_salary_avg_eur':
        if income_variable in ranked_cities.columns:
            ranked_cities[income_variable] = ranked_cities[income_variable].apply(lambda x: f"{x:.2f}€")
    


    # Round satisfaction variables to two decimal places
    for var in selected_satisfaction_vars:
        if var in ranked_cities.columns:
            ranked_cities[var] = ranked_cities[var].round(2)


    # Return the ranked cities for the table
    return ranked_cities[['Rank', 'City', 'Country Name', 'Score'] + columns_to_include].to_dict('records')




# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
