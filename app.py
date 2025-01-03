import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State, ALL
import dash_table
from dash_table.Format import Group
import plotly.express as px
import json
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import numpy as np


cities = pd.read_csv('data/cities.csv')
with open("data/interests.json", "r") as f:
    interest_categories = json.load(f)


# List of columns for which NaN values should be filled with country-wise averages
columns_to_fill = [
    'public_transport_aggregated',
    'education_facilities_aggregated',
    'healthcare_services_aggregated',
    'green_spaces_aggregated',
    'sports_facilities_aggregated',
    'cultural_facilities_aggregated',
    'job_availability_aggregated',
    'foreigner_integration_aggregated',
    'housing_affordability_aggregated',
    'admin_efficiency_aggregated',
    'transport_affordability_aggregated',
    'transport_accessibility_aggregated',
    'transport_frequency_aggregated',
    'transport_reliability_aggregated',
    'ticket_one_way_eur',
    'gasoline_liter_eur',
    'fitness_club_monthly_eur',
    'rent_1br_city_center_eur',
    'rent_1br_outside_center_eur',
    'rent_3br_city_center_eur',
    'rent_3br_outside_center_eur',
    'net_salary_avg_eur',
    'utilities_basic_eur',
    'mobile_phone_plan_eur',
    'preschool_monthly_eur',
    'price_sqm_city_center_eur'
]

# Fill NaN values in each column with the country-wise average
for col in columns_to_fill:
    cities[col] = cities.groupby('Country Name')[col].transform(lambda x: x.fillna(x.mean()))

# Define the columns to normalize (columns 36 to 120 inclusive)
columns_to_normalize = cities.columns[36:121]

# Step 1: Convert the specified columns to numeric, coercing errors to NaN
cities[columns_to_normalize] = cities[columns_to_normalize].apply(pd.to_numeric, errors='coerce')

# Step 2: Calculate the row-wise sum for the selected columns
row_sums = cities[columns_to_normalize].sum(axis=1)

# Step 3: Replace any row sums that are zero with one to avoid division by zero
row_sums = row_sums.replace(0, 1)

# Step 4: Normalize each cell by dividing by the row sum
cities[columns_to_normalize] = cities[columns_to_normalize].div(row_sums, axis=0)

# (Optional) Step 5: Fill any remaining NaN values with zero (if desired)
cities[columns_to_normalize] = cities[columns_to_normalize].fillna(0)


# External stylesheets
external_stylesheets = [
    dbc.themes.BOOTSTRAP,
    "https://use.fontawesome.com/releases/v5.8.1/css/all.css"
]





app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

# Label mappings
label_mapping = {
    'ticket_one_way_eur': 'Public Transport Price (One Way Ticket)',
    'gasoline_liter_eur': 'Gasoline Price per Liter',
    'fitness_club_monthly_eur': 'Fitness Club Fee (Monthly)',
    'rent_1br_city_center_eur': 'Rent Price 1 Bedroom Apartment City Center',
    'rent_1br_outside_center_eur': 'Rent Price 1 Bedroom Apartment Outside City Center',
    'rent_3br_city_center_eur': 'Rent Price 3 Bedroom Apartment City Center',
    'rent_3br_outside_center_eur': 'Rent Price 3 Bedroom Apartment Outside City Center',
    'net_salary_avg_eur': 'Avg. Net Salary Income',
    'utilities_basic_eur': 'Avg. Cost Apartment Utilities (Water, Electricity, Gas, ...)',
    'mobile_phone_plan_eur': 'Price Mobile Phone Plan',
    'preschool_monthly_eur': 'Price Monthly Preschool Fee',
    'price_sqm_city_center_eur': 'Buying Apartment (m² Price)'
}

cost_vars = [
    'rent_1br_city_center_eur',
    'rent_3br_city_center_eur',
    'rent_1br_outside_center_eur',
    'rent_3br_outside_center_eur',
    'utilities_basic_eur',
    'price_sqm_city_center_eur',
    'ticket_one_way_eur',
    'gasoline_liter_eur',
    'fitness_club_monthly_eur',
    'mobile_phone_plan_eur'
]

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
label_mapping.update(satisfaction_vars) 

# Layout
app.layout = html.Div([
    # Location component for client-side callbacks
    dcc.Location(id='url', refresh=True),
    
    # Add the Store Component to Keep Track of the Current View
    dcc.Store(id='page-view', data='main'),
    
    # Title Section with Background Color
    html.Div([
        html.H1(
            "European City Comparison Tool",
            style={
                'textAlign': 'center',
                'fontSize': '36px',
                'margin': '10px 0',
                'fontWeight': 'bold'
            }
        ),
        html.H3(
            "Find your ideal European city based on what matters most to you.",
            style={
                'textAlign': 'center',
                'fontStyle': 'italic',
                'margin': '0 0 10px 0',
                'fontSize': '20px'
            }
        ),
        html.P(
            "Use this application to explore and compare European cities according to your preferences. Adjust the filters and weights to prioritize factors like cost of living, income, satisfaction with urban services, and social interests. Data sources include Numbeo, Eurostat, and Facebook Marketing API.",
            style={
                'textAlign': 'center',
                'fontSize': '16px',
                'margin': '0 20px'
            }
        ),
        # Add the "About" Button Below the Paragraph
        html.Div([
            html.Button(
                "About This Tool",
                id='about-button',
                n_clicks=0,
                style={
                    'display': 'block',
                    'margin': '20px auto',
                    'padding': '10px 20px',
                    'fontSize': '18px',
                    'backgroundColor': '#007BFF',
                    'color': 'white',
                    'border': 'none',
                    'borderRadius': '5px',
                    'cursor': 'pointer'
                }
            )
        ]),
        # Add the About Content Div Below the Title Section
        html.Div([
            html.Div(id='about-content', children=[], style={
                'display': 'none',  # Initially hidden
                'padding': '20px',
                'backgroundColor': '#fff',
                'borderRadius': '5px',
                'boxShadow': '0 0 10px rgba(0,0,0,0.1)',
                'marginTop': '20px'
            })
        ])
    ], style={
        'backgroundColor': '#e6f7ff',
        'padding': '20px 0',
        'marginLeft': '30%',  
        'borderBottom': '2px solid #ccc'
    }),

    # Main Content: Sidebar and Main Panel
    html.Div([
        # Sidebar Menu on the Left
        html.Div([
            html.Br(),
            html.H2("Customize Your Preferences", style={
                "textAlign": "center",
                'fontSize': '28px',
                'fontWeight': 'bold',
                'marginBottom': '20px'
            }),

            # Population Filter Section
            html.Div([
                html.H4("Population", style={'fontSize': '20px', 'fontWeight': 'bold'}),
                html.Label([
                    'Population Range',
                    html.I(className="fas fa-info-circle", id='population-tooltip', style={
                        'margin-left': '8px',
                        'cursor': 'pointer',
                        'color': '#007BFF'
                    })
                ], style={'fontSize': '16px', 'fontWeight': 'bold'}),
                dbc.Tooltip(
                    "Use the slider to select cities with populations within your desired range.",
                    target='population-tooltip',
                    placement='right',
                ),
                html.P("From 0 to 15M+", style={'fontSize': '14px', 'marginTop': '5px'}),
                html.Div(
                    dcc.RangeSlider(
                        id='population-slider',
                        min=0,
                        max=10,
                        marks={
                            0: '0',
                            1: '100k',
                            2: '200k',
                            3: '300k',
                            4: '400k',
                            5: '500k',
                            6: '750k',
                            7: '1M',
                            8: '3M',
                            9: '5M',
                            10: '15M+'
                        },
                        value=[0, 10],
                        step=None,
                        allowCross=False,
                        tooltip={"placement": "bottom", "always_visible": False}
                    ),
                    style={'width': '100%', 'margin': '0 auto'}  # Apply the style here
                ),
            ], style={'marginBottom': '30px'}),

            # Country Filter Section
            html.Div([
                html.H4("Country", style={'fontSize': '20px', 'fontWeight': 'bold'}),
                html.Label([
                    'Select Countries',
                    html.I(className="fas fa-info-circle", id='country-tooltip', style={
                        'margin-left': '8px',
                        'cursor': 'pointer',
                        'color': '#007BFF'
                    })
                ], style={'fontSize': '16px', 'fontWeight': 'bold'}),
                dbc.Tooltip(
                    "Select one or more countries to include cities only from those countries. Select 'All Countries' to include all.",
                    target='country-tooltip',
                    placement='right',
                ),
                dcc.Dropdown(
                    id='country-dropdown',
                    options=[{'label': 'All Countries', 'value': 'All'}] + [{'label': country, 'value': country} for country in sorted(cities['Country Name'].unique())],
                    value=['All'],  # Default to 'All Countries'
                    multi=True,
                    placeholder="Select Country"
                ),
            ], style={'marginBottom': '30px'}),

            # City Filter Section
            html.Div([
                html.H4("City Selection", style={'fontSize': '20px', 'fontWeight': 'bold'}),
                html.Label([
                    'Filter by City',
                    html.I(className="fas fa-info-circle", id='city-tooltip', style={
                        'margin-left': '8px',
                        'cursor': 'pointer',
                        'color': '#007BFF'
                    })
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
            ], style={'marginBottom': '30px'}),

            # Income Variable Section
            html.Div([
                html.H4("Income", style={'fontSize': '20px', 'fontWeight': 'bold'}),
                html.Label([
                    'Income Variable',
                    html.I(className="fas fa-info-circle", id='income-tooltip', style={
                        'margin-left': '8px',
                        'cursor': 'pointer',
                        'color': '#007BFF'
                    })
                ], style={'fontSize': '16px', 'fontWeight': 'bold'}),
                dbc.Tooltip(
                    "Include average net salary in the comparison.",
                    target='income-tooltip',
                    placement='right',
                ),
                html.P("From 1 to 5, how important is Income to you?", style={'fontSize': '14px', 'marginTop': '5px'}),
                dcc.Dropdown(
                    id='income-dropdown',
                    options=[{'label': label_mapping['net_salary_avg_eur'], 'value': 'net_salary_avg_eur'}],
                    value=None,  
                    placeholder="Select Income Variable"
                ),
                dcc.Slider(
                    id='income-slider',
                    min=1, max=5, step=1, value=1,
                    marks={i: str(i) for i in range(1, 6)},
                    tooltip={"placement": "bottom", "always_visible": False}
                ),
            ], style={'marginBottom': '30px'}),

            # Cost of Living Variables Section
            html.Div([
                html.H4("Cost of Living", style={'fontSize': '20px', 'fontWeight': 'bold'}),
                html.Label([
                    "Cost of Living Variables",
                    html.I(className="fas fa-info-circle", id='cost-tooltip', style={
                        'margin-left': '8px',
                        'cursor': 'pointer',
                        'color': '#007BFF'
                    })
                ], style={'fontSize': '16px', 'fontWeight': 'bold'}),
                dbc.Tooltip(
                    "Select cost of living factors to include in the comparison. Click 'Add Variable' to select more.",
                    target='cost-tooltip',
                    placement='right',
                ),
                html.P("From 1 to 5, how important is minimizing Cost of Living to you?", style={'fontSize': '14px', 'marginTop': '5px'}),
                html.Button('Add Variable', id='add-button', n_clicks=0,
                        style={
                            'marginTop': '10px',
                            'marginBottom': '10px',
                            'backgroundColor': '#28a745',  # Changed to green
                            'color': 'white',
                            'border': 'none',
                            'padding': '8px 12px',
                            'borderRadius': '4px',
                            'cursor': 'pointer'
                        }),

                html.Div(id='cost-variable-dropdowns', children=[]),
            ], style={'marginBottom': '30px', 'borderBottom': '1px solid #ccc', 'paddingBottom': '20px'}),

            # Urban Livability Variables Section
            html.Div([
                html.H4("Urban Livability", style={'fontSize': '20px', 'fontWeight': 'bold'}),
                html.Label([
                    "Urban Livability Variables",
                    html.I(className="fas fa-info-circle", id='satisfaction-tooltip', style={
                        'margin-left': '8px',
                        'cursor': 'pointer',
                        'color': '#007BFF'
                    })
                ], style={'fontSize': '16px', 'fontWeight': 'bold'}),
                dbc.Tooltip(
                    "Select urban livability factors to include in the comparison. Click 'Add Variable' to select more and adjust their importance.",
                    target='satisfaction-tooltip',
                    placement='right',
                ),
                html.P("From 1 to 5, how important is the selected Urban Livability variable to you?", style={'fontSize': '14px', 'marginTop': '5px'}),
                html.Button('Add Variable', id='add-satisfaction-button', n_clicks=0,
                            style={
                                'marginTop': '10px',
                                'marginBottom': '10px',
                                'backgroundColor': '#28a745',
                                'color': 'white',
                                'border': 'none',
                                'padding': '8px 12px',
                                'borderRadius': '4px',
                                'cursor': 'pointer'
                            }),
                html.Div(id='satisfaction-variable-dropdowns', children=[]),
            ], style={'marginBottom': '30px', 'borderBottom': '1px solid #ccc', 'paddingBottom': '20px'}),

            # Personal Interests Section (Unified)
            html.Div([
                html.H4("Personal Interests", style={'fontSize': '20px', 'fontWeight': 'bold'}),
                html.Label([
                    "Select your Personal Interests",
                    html.I(className="fas fa-info-circle", id='fb-tooltip', style={
                        'margin-left': '8px',
                        'cursor': 'pointer',
                        'color': '#007BFF'
                    })
                ], style={'fontSize': '16px', 'fontWeight': 'bold'}),
                dbc.Tooltip(
                    "Select categories you are interested in to find those cities with the highest number of people interested in the same categories on Meta services.",
                    target='fb-tooltip',
                    placement='right',
                ),
                html.Div([
                    dbc.DropdownMenu(
                        label=category,
                        children=[
                            dbc.Checklist(
                                options=[
                                    {'label': interest, 'value': interest}
                                    for interest in interests
                                ],
                                value=[],
                                id={'type': 'fb-interests', 'index': category},
                                style={'margin-left': '20px', 'margin-bottom': '5px'}
                            )
                        ],
                        direction='down',
                        style={
                            'marginBottom': '10px',
                            'width': '250px',  # Set a fixed width to unify
                        }
                    )
                    for category, interests in interest_categories.items()
                ], style={
                    'padding': '10px',
                    'border': '1px solid #ccc',
                    'borderRadius': '5px',
                    'backgroundColor': '#fff'
                }),
                html.P("From 1 to 5, how important are communities with similar interests to you?", style={'fontSize': '14px', 'marginTop': '10px'}),
                dcc.Slider(
                    id='social-fit-slider',
                    min=1, max=5, step=1, value=1,
                    marks={i: str(i) for i in range(1, 6)},
                    tooltip={"placement": "bottom", "always_visible": False}
                ),
            ], style={'marginBottom': '30px'}),

            html.Br(),
            html.Br(),
            html.Br(),
            html.Br(),

            # Reset Button at the Very Bottom
            html.Div([
                html.Button(
                    "Reset Settings",
                    id='reset-button',
                    n_clicks=0,
                    style={
                        'width': '100%',
                        'padding': '12px',
                        'backgroundColor': '#ff4d4f',
                        'color': 'white',
                        'border': 'none',
                        'borderRadius': '5px',
                        'cursor': 'pointer',
                        'fontSize': '16px',
                        'fontWeight': 'bold'
                    }
                )
            ], style={'marginTop': '20px'}),
        ], style={
            'padding': '30px',
            'width': '25%',
            'backgroundColor': '#f8f9fa',  # Slightly lighter grey for better contrast
            'boxSizing': 'border-box',
            'overflowY': 'auto',
            'height': '100vh',
            'position': 'fixed',
            'left': '0',
            'top': '0',
            'bottom': '0',
            'zIndex': '1',
            'borderRight': '2px solid #dee2e6'
        }),

        # Main Panel on the Right
        html.Div([
            # Map Section
            html.Div([
                html.H4([
                    "Map of Cities",
                    html.I(className="fas fa-info-circle", id='map-tooltip', style={
                        'margin-left': '8px',
                        'cursor': 'pointer',
                        'color': '#007BFF'
                    })
                ], style={'fontSize': '22px', 'fontWeight': 'bold'}),
                dbc.Tooltip(
                    "This map shows the cities based on your preferences. Hover over markers for more info.",
                    target='map-tooltip',
                    placement='right',
                ),
                dcc.Graph(
                    id="city-map",
                    style={'height': '600px'},
                    config={'scrollZoom': False}  # Reverted to original configuration
                ),
            ], style={'padding': '20px', 'backgroundColor': '#fff', 'borderRadius': '5px', 'boxShadow': '0 0 10px rgba(0,0,0,0.1)'}),

            html.Br(),

            # Ranking Table
            html.Div([
                html.H4([
                    "City Rankings",
                    html.I(className="fas fa-info-circle", id='ranking-tooltip', style={
                        'margin-left': '8px',
                        'cursor': 'pointer',
                        'color': '#007BFF'
                    })
                ], style={'fontSize': '22px', 'fontWeight': 'bold'}),
                dbc.Tooltip(
                    "This table lists cities ranked based on your selected preferences and weights.",
                    target='ranking-tooltip',
                    placement='right',
                ),
                dash_table.DataTable(
                    id='ranking-table',
                    columns=[
                        {'name': 'Rank', 'id': 'Rank'},
                        {'name': 'City', 'id': 'City'},
                        {'name': 'Country', 'id': 'Country Name'},
                        {'name': 'Score', 'id': 'Score'}
                    ],
                    style_table={
                        'overflowX': 'auto',
                        'width': '100%',
                        'backgroundColor': '#f8f9fa',
                        'borderRadius': '5px',
                        'boxShadow': '0 0 10px rgba(0,0,0,0.1)'
                    },
                    style_cell={
                        'textAlign': 'left',
                        'backgroundColor': '#f8f9fa',
                        'padding': '10px',
                        'fontSize': '16px'
                    },
                    style_header={
                        'backgroundColor': '#007BFF',
                        'color': 'white',
                        'fontWeight': 'bold',
                        'fontSize': '16px'
                    },
                    sort_action='native',
                    page_action='native',
                    page_current=0,
                    page_size=20  # Display 20 cities per page
                ),
            ], style={'padding': '20px', 'backgroundColor': '#fff', 'borderRadius': '5px', 'boxShadow': '0 0 10px rgba(0,0,0,0.1)'}),
        ], style={
            'marginLeft': '25%',  # Offset to match the sidebar
            'width': '75%',       # Occupy the remaining width
            'padding': '30px',
            'backgroundColor': '#f1f3f5',  # Light blue-grey background for the main content
            'boxSizing': 'border-box'
        }),
    ], style={'display': 'flex', 'flexDirection': 'row'}),

    # Hidden div to trigger page reload and scrolling
    html.Div(id='dummy-output', style={'display': 'none'})
])

# Population thresholds for the slider
population_thresholds = [
    0,
    100_000,
    200_000,
    300_000,
    400_000,
    500_000,
    750_000,
    1_000_000,
    3_000_000,
    5_000_000,
    15_000_000
]

@app.callback(
    Output('country-dropdown', 'value'),
    Input('country-dropdown', 'value')
)
def update_country_dropdown(selected_countries):
    if selected_countries is None:
        raise dash.exceptions.PreventUpdate
    if 'All' in selected_countries and len(selected_countries) > 1:
        selected_countries.remove('All')
    elif not selected_countries:
        selected_countries = ['All']
    return selected_countries

@app.callback(
    Output({'type': 'dynamic-dropdown', 'index': ALL}, 'options'),
    Input({'type': 'dynamic-dropdown', 'index': ALL}, 'value'),
)
def update_dynamic_dropdown_options(selected_values):
    options = []
    selected_vars = [v for v in selected_values if v is not None]
    for i, v in enumerate(selected_values):
        available_vars = [{'label': label_mapping[var], 'value': var}
                          for var in cost_vars if var not in selected_vars or var == v]
        options.append(available_vars)
    return options

@app.callback(
    Output({'type': 'dynamic-satisfaction-dropdown', 'index': ALL}, 'options'),
    Input({'type': 'dynamic-satisfaction-dropdown', 'index': ALL}, 'value'),
)
def update_satisfaction_dropdown_options(selected_values):
    options = []
    selected_vars = [v for v in selected_values if v is not None]
    for i, v in enumerate(selected_values):
        available_vars = [{'label': label_mapping[var], 'value': var}
                          for var in satisfaction_vars if var not in selected_vars or var == v]
        options.append(available_vars)
    return options

# Callback to update the city dropdown based on the selected country and population range
@app.callback(
    Output('city-dropdown', 'options'),
    [Input('country-dropdown', 'value'), Input('population-slider', 'value')]
)
def update_city_dropdown(selected_countries, population_value):
    # Convert slider value into population ranges
    def get_population_value(slider_value):
        return population_thresholds[int(slider_value)]
    
    population_min = get_population_value(population_value[0])
    population_max = get_population_value(population_value[1])

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
    available_vars = [{'label': label_mapping[var], 'value': var} for var in cost_vars if var not in selected_vars]

    if not available_vars:  # If no more variables are available, don't add new dropdowns
        return children

    # Define the new dropdown and slider for the new cost variable
    new_element = html.Div([
        dcc.Dropdown(
            id={'type': 'dynamic-dropdown', 'index': n_clicks},
            options=available_vars,  # Filtered options for cost variables
            value=None,  # No default selected
            placeholder="Select Cost Variable"
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
    available_vars = [{'label': label_mapping[var], 'value': var} for var in satisfaction_vars if var not in selected_vars]

    if not available_vars:  # If no more variables are available, don't add new dropdowns
        return children

    # Define the new dropdown and slider for the new satisfaction variable
    new_element = html.Div([
        dcc.Dropdown(
            id={'type': 'dynamic-satisfaction-dropdown', 'index': n_clicks},
            options=available_vars,  # Filtered options for satisfaction variables
            value=None,  # No default selected
            placeholder="Select Satisfaction Variable"
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
        Input({'type': 'fb-interests', 'index': ALL}, 'value'),
        State({'type': 'fb-interests', 'index': ALL}, 'id'),
        Input('social-fit-slider', 'value')
    ]
)
def update_map(population_value, selected_countries, selected_cities,
               selected_variables, variable_weights,
               income_variable, income_weight,
               selected_satisfaction_vars, satisfaction_weights,
               fb_interest_values, fb_interest_ids,
               social_fit_weight):

    # Flatten the list of selected FB interests
    selected_fb_interests = []
    for interests in fb_interest_values:
        selected_fb_interests.extend(interests)

    # Convert slider value into population ranges
    def get_population_value(slider_value):
        return population_thresholds[int(slider_value)]

    population_min = get_population_value(population_value[0])
    population_max = get_population_value(population_value[1])

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
    selected_cost_vars = [var for var in selected_variables if var is not None]
    selected_satisfaction_vars = [var for var in selected_satisfaction_vars if var is not None]
    variable_weights = variable_weights[:len(selected_cost_vars)]
    satisfaction_weights = satisfaction_weights[:len(selected_satisfaction_vars)]

    # Add income variable if selected
    if income_variable is not None:
        selected_cost_vars.append(income_variable)
        variable_weights.append(income_weight)

    # Prepare normalization for variables (min-max scaling)
    scaler = MinMaxScaler()

    weighted_scores = []

    # Handle cost of living variables
    reverse_vars = [
            'rent_1br_city_center_eur',
            'rent_3br_city_center_eur',
            'rent_1br_outside_center_eur',
            'rent_3br_outside_center_eur',
            'utilities_basic_eur',
            'price_sqm_city_center_eur',
            'ticket_one_way_eur',
            'gasoline_liter_eur',
            'fitness_club_monthly_eur',
            'mobile_phone_plan_eur'
    ]

    # Normalize and weight cost of living variables
    for idx, var in enumerate(selected_cost_vars):
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
    if selected_fb_interests:
        fb_interest_columns = [col for col in selected_fb_interests if col in filtered_cities.columns]
        
        if fb_interest_columns:
            # Normalize each selected interest category
            normalized_interest_scores = pd.DataFrame(index=filtered_cities.index)
            for col in fb_interest_columns:
                data = filtered_cities[[col]].astype(float)
                normalized = scaler.fit_transform(data)
                normalized_interest_scores[col] = normalized.flatten()
            if not normalized_interest_scores.empty:
                avg_normalized_score = normalized_interest_scores.mean(axis=1)
                # Multiply by 100 to get percentage for display
                interest_fit_display = pd.Series((avg_normalized_score * 100).round(2), index=filtered_cities.index)
                filtered_cities['Interest Fit Display'] = interest_fit_display.astype(str) + '%'
                # For scoring, use the average normalized score
                avg_normalized_score = avg_normalized_score.values.reshape(-1, 1)
                weighted_scores.append(avg_normalized_score * social_fit_weight)
            else:
                filtered_cities['Interest Fit Display'] = ''
        else:
            filtered_cities['Interest Fit Display'] = ''
    else:
        filtered_cities['Interest Fit Display'] = ''

    # Calculate final score only if there are any selected variables
    if weighted_scores:
        total_weighted_scores = np.sum(weighted_scores, axis=0)
        # Sum across all variables to get a single score per city
        filtered_cities['Score'] = total_weighted_scores.sum(axis=1)
        filtered_cities['Score'] = filtered_cities['Score'].round(3)
    else:
        filtered_cities['Score'] = 0  # Set score to 0 if no variables are selected

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
            cmin=filtered_cities['Score'].min() if (filtered_cities['Score'] > 0).any() else 0,
            cmax=filtered_cities['Score'].max() if (filtered_cities['Score'] > 0).any() else 1,
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
            "Population: %{customdata[1]:,}<br>" +
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


# Callback to update the ranking table columns
@app.callback(
    Output('ranking-table', 'columns'),
    [
        Input({'type': 'dynamic-dropdown', 'index': ALL}, 'value'),
        Input({'type': 'dynamic-satisfaction-dropdown', 'index': ALL}, 'value'),
        Input('income-dropdown', 'value'),
        Input({'type': 'fb-interests', 'index': ALL}, 'value'),
    ]
)
def update_table_columns(selected_variables, selected_satisfaction_vars, income_variable,
                         fb_interest_values):
    base_columns = [
        {'name': 'Rank', 'id': 'Rank'},
        {'name': 'City', 'id': 'City'},
        {'name': 'Country', 'id': 'Country Name'},
        {'name': 'Score', 'id': 'Score'}
    ]
    
    # Include the income variable in the list of columns
    all_selected_vars = [var for var in selected_variables if var] + [var for var in selected_satisfaction_vars if var]
    if income_variable is not None:
        all_selected_vars.append(income_variable)
    
    # Prepare variable columns
    variable_columns = []
    for var in all_selected_vars:
        if var in satisfaction_vars:
            display_name = label_mapping.get(var, var)
            variable_columns.append({'name': display_name, 'id': display_name})
        else:
            display_name = label_mapping.get(var, var)
            variable_columns.append({'name': display_name, 'id': var})
    
    # Determine if any FB interest categories are selected
    selected_fb_interests = []
    for interests in fb_interest_values:
        selected_fb_interests.extend(interests)
    if selected_fb_interests:
        variable_columns.append({'name': 'Interest Fit', 'id': 'Interest Fit Display'})

    return base_columns + variable_columns


# Callback to update the ranking table data
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
        Input({'type': 'fb-interests', 'index': ALL}, 'value'),
        State({'type': 'fb-interests', 'index': ALL}, 'id'),
        Input('social-fit-slider', 'value')
    ]
)
def update_ranking_table(population_value, selected_countries, selected_cities,
                         selected_variables, variable_weights,
                         income_variable, income_weight,
                         selected_satisfaction_vars, satisfaction_weights,
                         fb_interest_values, fb_interest_ids,
                         social_fit_weight):

    # Flatten the list of selected FB interests
    selected_fb_interests = []
    for interests in fb_interest_values:
        selected_fb_interests.extend(interests)

    # Convert slider value into population ranges
    def get_population_value(slider_value):
        return population_thresholds[int(slider_value)]

    population_min = get_population_value(population_value[0])
    population_max = get_population_value(population_value[1])

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
    selected_cost_vars = [var for var in selected_variables if var is not None]
    selected_satisfaction_vars = [var for var in selected_satisfaction_vars if var is not None]
    variable_weights = variable_weights[:len(selected_cost_vars)]
    satisfaction_weights = satisfaction_weights[:len(selected_satisfaction_vars)]

    # Add income variable if selected
    if income_variable is not None:
        selected_cost_vars.append(income_variable)
        variable_weights.append(income_weight)

    # Prepare normalization for variables (min-max scaling)
    scaler = MinMaxScaler()

    weighted_scores = []

    # Handle cost of living variables
    reverse_vars = [
            'rent_1br_city_center_eur',
            'rent_3br_city_center_eur',
            'rent_1br_outside_center_eur',
            'rent_3br_outside_center_eur',
            'utilities_basic_eur',
            'price_sqm_city_center_eur',
            'ticket_one_way_eur',
            'gasoline_liter_eur',
            'fitness_club_monthly_eur',
            'mobile_phone_plan_eur'
    ]

    # Normalize and weight cost of living variables
    for idx, var in enumerate(selected_cost_vars):
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
    if selected_fb_interests:
        fb_interest_columns = [col for col in selected_fb_interests if col in filtered_cities.columns]
        
        if fb_interest_columns:
            # Normalize each selected interest category
            normalized_interest_scores = pd.DataFrame(index=filtered_cities.index)
            for col in fb_interest_columns:
                data = filtered_cities[[col]].astype(float)
                normalized = scaler.fit_transform(data)
                normalized_interest_scores[col] = normalized.flatten()
            if not normalized_interest_scores.empty:
                avg_normalized_score = normalized_interest_scores.mean(axis=1)
                # Multiply by 100 to get percentage for display
                interest_fit_display = pd.Series((avg_normalized_score * 100).round(2), index=filtered_cities.index)
                filtered_cities['Interest Fit Display'] = interest_fit_display.astype(str) + '%'
                # For scoring, use the average normalized score
                avg_normalized_score = avg_normalized_score.values.reshape(-1, 1)
                weighted_scores.append(avg_normalized_score * social_fit_weight)
            else:
                filtered_cities['Interest Fit Display'] = ''
        else:
            filtered_cities['Interest Fit Display'] = ''
    else:
        filtered_cities['Interest Fit Display'] = ''

    # Calculate final score only if there are any selected variables
    if weighted_scores:
        total_weighted_scores = np.sum(weighted_scores, axis=0)
        # Sum across all variables to get a single score per city
        filtered_cities['Score'] = total_weighted_scores.sum(axis=1)
        # Round Score to two decimal places
        filtered_cities['Score'] = filtered_cities['Score'].round(2)
    else:
        filtered_cities['Score'] = 0  # Set score to 0 if no variables are selected

    # Assign quartile-based labels to satisfaction variables
    for var in selected_satisfaction_vars:
        if var in filtered_cities.columns:
            # Calculate quartiles
            quartiles = filtered_cities[var].quantile([0.25, 0.5, 0.75]).values
            # Assign labels based on quartiles
            filtered_cities[var + '_label'] = pd.cut(
                filtered_cities[var],
                bins=[-np.inf, quartiles[0], quartiles[1], quartiles[2], np.inf],
                labels=["Bad", "Medium", "Good", "Very Good"]
            )

    # Combine original values with scores and rank
    columns_to_include = selected_cost_vars + selected_satisfaction_vars
    if 'Interest Fit Display' in filtered_cities.columns:
        columns_to_include.append('Interest Fit Display')

    # Create a copy for ranking to avoid SettingWithCopyWarning
    ranked_cities = filtered_cities[['City', 'Country Name', 'Score']].copy()

    # Add selected cost of living variables with "€" appended
    for var in selected_cost_vars:
        if var in filtered_cities.columns:
            # Round to two decimal places and format with '€'
            ranked_cities[var] = filtered_cities[var].round(2).apply(lambda x: f"{x:.2f}€")

    # Add selected satisfaction variables with labels
    for var in selected_satisfaction_vars:
        if var + '_label' in filtered_cities.columns:
            display_name = label_mapping.get(var, var)
            ranked_cities[display_name] = filtered_cities[var + '_label']

    # Add Interest Fit Display if applicable
    if 'Interest Fit Display' in filtered_cities.columns:
        ranked_cities['Interest Fit Display'] = filtered_cities['Interest Fit Display']

    # Add Rank based on Score
    ranked_cities['Rank'] = ranked_cities['Score'].rank(method='min', ascending=False)
    ranked_cities = ranked_cities.sort_values(by='Rank')  # No longer limit to top 20

    # Return the ranked cities for the table
    return ranked_cities[['Rank', 'City', 'Country Name', 'Score'] + 
                        [var for var in selected_cost_vars if var != 'net_salary_avg_eur'] + 
                        ([income_variable] if income_variable else []) + 
                        [label_mapping.get(var, var) for var in selected_satisfaction_vars] + 
                        (['Interest Fit Display'] if 'Interest Fit Display' in ranked_cities.columns else [])].to_dict('records')




# Callback to toggle the About Content
@app.callback(
    [Output('about-content', 'children'),
     Output('about-content', 'style')],
    Input('about-button', 'n_clicks'),
    State('about-content', 'children'),
    prevent_initial_call=True
)
def toggle_about(n_clicks, current_children):
    if n_clicks is None or n_clicks == 0:
        # If the button hasn't been clicked, keep the Div hidden
        return "", {'display': 'none'}
    elif n_clicks % 2 == 1:
        # On odd clicks, display the About content
        about_text = [
            html.H2("About This Tool", style={'textAlign': 'center', 'color': '#2c3e50'}),
            
            # Purpose and Overview
            html.H4("Purpose and Overview", style={'marginTop': '20px', 'color': '#34495e'}),
            html.P(
                "The European City Comparison Tool is designed to help users find their ideal European city based on personalized preferences. "
                "By adjusting various filters and weights, users can prioritize factors such as cost of living, income, urban satisfaction metrics, and social interests to generate a tailored ranking of cities."
            ),
            
            # Data Sources
            html.H4("Data Sources", style={'marginTop': '20px', 'color': '#34495e'}),
            html.Ul([
                html.Li([
                    html.Strong("Numbeo.com: "),
                    "Provides objective cost-of-living parameters including rent prices, utilities, transportation costs, and average net salaries. These data correspond to the 'Income Variable' and 'Cost of Living Variables' in the settings panel."
                ]),
                html.Li([
                    html.Strong("Eurostat's Urban Audit: "),
                    "Offers subjective quality of life measures based on resident satisfaction surveys covering various urban services and amenities. These metrics are linked to the 'Urban Livability Variables' in the settings panel."
                ]),
                html.Li([
                    html.Strong("Facebook Marketing API: "),
                    "Supplies social media interest data, allowing analysis of residents' personal interests and community dynamics. These interests are utilized in the 'Personal Interests' section of the settings panel."
                ]),
            ]),
            
            # Data Normalization
            html.H4("Data Normalization", style={'marginTop': '20px', 'color': '#34495e'}),
            html.P(
                "To ensure comparability across different data sources and scales, all variables undergo a normalization process using Min-Max scaling. "
                "This transformation scales each variable to a range between 0 and 1, allowing for meaningful aggregation and comparison. "
                "For cost-related metrics where lower values are preferable (e.g., rent prices), the scale is inverted so that lower costs correspond to higher scores."
            ),
            
            # Scoring Methodology
            html.H4("Scoring Methodology", style={'marginTop': '20px', 'color': '#34495e'}),
            html.P(
                "The scoring system aggregates multiple Quality of Life (QoL) parameters into a single score for each city based on user preferences. Here's how it works:"
            ),
            html.Ol([
                html.Li([
                    html.Strong("Selection of Parameters: "),
                    "Users select various QoL factors such as countries, cities, income variables, cost of living factors, urban livability metrics, and personal interests from the sidebar settings."
                ]),
                html.Li([
                    html.Strong("Assigning Weights: "),
                    "Each selected parameter can be assigned a weight between 1 (least important) and 5 (most important) to reflect its significance to the user."
                ]),
                html.Li([
                    html.Strong("Weighted Aggregation: "),
                    "Each parameter's normalized score is multiplied by its assigned weight. The sum of these weighted scores results in an overall QoL score for the city."
                ]),
                html.Li([
                    html.Strong("Interest Fit Score: "),
                    "Social interests are analyzed using the Facebook Marketing API to determine the proportion of residents interested in specific categories. These interest fit scores are averaged and weighted to produce an 'Interest Fit' percentage, indicating how well a city's social profile matches the user's preferences."
                ]),
            ]),
            
            # Social Interests Detailed Explanation
            html.H4("Understanding Social Interests", style={'marginTop': '20px', 'color': '#34495e'}),
            html.P(
                "Social interests refer to the personal and cultural preferences of a city's residents, such as hobbies, activities, and areas of passion. These interests are measured using the Facebook Marketing API, which estimates the number of Facebook users in each city interested in predefined categories."
            ),
            html.P(
                "Here's how social interests are integrated into the tool:"
            ),
            html.Ul([
                html.Li([
                    html.Strong("Data Collection: "),
                    "The Facebook Marketing API collects aggregated and anonymized data on user interests based on their online interactions, likes, shares, and content engagement within specific geographic regions."
                ]),
                html.Li([
                    html.Strong("Normalization: "),
                    "Interest data for each city is normalized to account for differences in city populations and Facebook user bases. This ensures that interest proportions are comparable across cities of varying sizes."
                ]),
                html.Li([
                    html.Strong("Scoring: "),
                    "Normalized interest scores are averaged and weighted according to user-defined preferences, contributing to the overall QoL score. This allows the tool to reflect not only economic and infrastructural factors but also the cultural and social fabric that enhances urban living."
                ]),
            ]),
            

        ]
        return about_text, {
            'display': 'block',
            'padding': '30px',
            'backgroundColor': '#f9f9f9',
            'borderRadius': '8px',
            'boxShadow': '0 4px 8px rgba(0,0,0,0.1)',
            'marginTop': '20px',
            'fontFamily': 'Arial, sans-serif'
        }
    else:
        # On even clicks, hide the About content
        return "", {'display': 'none'}



# Client-Side Callback to Scroll to the About Section
app.clientside_callback(
    """
    function(n_clicks) {
        if (n_clicks && n_clicks % 2 === 1) {
            var element = document.getElementById('about-content');
            if (element) {
                element.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        }
        return "";
    }
    """,
    Output('dummy-output', 'children'),  # Dummy output to satisfy the callback
    Input('about-button', 'n_clicks')
)


# Client-Side Callback to Reload the Page on Reset Button Click
app.clientside_callback(
    """
    function(n_clicks) {
        if (n_clicks > 0) {
            window.location.reload();
        }
        return window.location.href;
    }
    """,
    Output('url', 'href'),
    Input('reset-button', 'n_clicks')
)




# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
