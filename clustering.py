import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.express as px

def run_clustering(df):
    # 1. Select Features
    cols = ['TotalWorkingYears', 'MonthlyIncome', 'JobSatisfaction']
    X = df[cols].copy()
    
    # 2. Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 3. Train K-Means (Random State 42 ensures we get the same groups every time)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    # 4. Assign Human-Readable Labels (Based on our Analysis)
    # Cluster 0 usually = Low Income / Low Sat
    # Cluster 1 usually = Low Income / High Sat
    # Cluster 2 usually = High Income
    
    df['Cluster'] = clusters
    
    # Create a new column with names instead of numbers
    cluster_names = {
        0: 'Disengaged Juniors (Risk)',
        1: 'Motivated Juniors (Safe)',
        2: 'Senior Veterans (Stable)'
    }
    df['Cluster Name'] = df['Cluster'].map(cluster_names)
    
    # 5. Create BIG 3D Plot
    # height=900 makes it very tall and clear
    fig = px.scatter_3d(df, 
                        x='TotalWorkingYears', 
                        y='MonthlyIncome', 
                        z='JobSatisfaction',
                        color='Cluster Name',
                        color_discrete_map={
                            'Disengaged Juniors (Risk)': 'purple',
                            'Motivated Juniors (Safe)': 'teal', 
                            'Senior Veterans (Stable)': 'gold'
                        },
                        opacity=0.8,
                        height=900,  # <--- This makes it BIG
                        width=1200,  # <--- This makes it WIDE
                        title="Employee Segments: The 'Happiness Layer' View"
                        )
    
    # Set initial camera angle to show the layers clearly
    fig.update_layout(scene_camera=dict(
        eye=dict(x=1.5, y=1.5, z=0.5)
    ))
    
    # Calculate Average Stats for the Table
    profile_table = df.groupby('Cluster Name')[cols].mean().round(2)
    
    return fig, profile_table