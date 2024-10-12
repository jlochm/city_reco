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


html.Br(),


# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
