import pandas as pd
import numpy as np
import re

def parse_formation(text):
    text = str(text).lower()
    if 'shotgun' in text: return 'Shotgun'
    if 'pistol' in text: return 'Pistol'
    if 'empty' in text: return 'Empty'
    if 'trips' in text: return 'Trips'
    if 'bunch' in text: return 'Bunch'
    if 'wildcat' in text: return 'Wildcat'
    return 'Standard' 

def parse_direction(text):
    text = str(text).lower()
    if 'left' in text: return 'Left'
    if 'right' in text: return 'Right'
    if 'middle' in text or 'center' in text: return 'Middle'
    return 'Unknown'

def normalize_play_type(play_type):
    play_type = str(play_type).lower()
    if 'rush' in play_type: return 'Run'
    if 'pass' in play_type or 'sack' in play_type or 'interception' in play_type: return 'Pass'
    return 'Other'

def recommend_cover_scheme(row):
    dist = row['Distance']
    yards_to_goal = row['YardsToGoal']
    
    if yards_to_goal <= 10:
        return 'Goal Line'
        
    if dist <= 3:
        return 'Cover 1' 
        
    if 3 < dist <= 7:
        return 'Cover 2' 
        
    if dist > 7:
        return 'Cover 3' 
        
    return 'Cover 2' 

def process_data(input_file, output_file):
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    
    df['play_type_clean'] = df['PlayType'].apply(normalize_play_type)
    
    df = df[df['play_type_clean'].isin(['Run', 'Pass'])]
    
    df['is_pass'] = (df['play_type_clean'] == 'Pass').astype(int)
    team_stats = df.groupby('Offense')['is_pass'].mean().reset_index()
    team_stats.columns = ['Offense', 'team_pass_rate']
    
    print("\nTop 5 Passing Teams:")
    print(team_stats.sort_values('team_pass_rate', ascending=False).head())
    
    team_stats.to_csv('team_stats.csv', index=False)
    print("Team stats saved.")
    
    df = df.merge(team_stats, on='Offense', how='left')
    
    df['offensive_formation'] = df['PlayText'].apply(parse_formation)
    df['play_direction'] = df['PlayText'].apply(parse_direction)
    
    start_len = len(df)
    df = df.dropna(subset=['Down', 'Distance', 'YardsToGoal', 'OffenseScore', 'DefenseScore', 'Period', 'Clock Minutes'])
    df['Down'] = pd.to_numeric(df['Down'], errors='coerce')
    df['Distance'] = pd.to_numeric(df['Distance'], errors='coerce')
    df['YardsToGoal'] = pd.to_numeric(df['YardsToGoal'], errors='coerce')
    
    df['score_diff'] = df['OffenseScore'] - df['DefenseScore']
    df['seconds_remaining'] = (4 - df['Period']) * 900 + df['Clock Minutes'] * 60 + df['Clock Seconds']
    
    df = df.dropna(subset=['Down', 'Distance', 'YardsToGoal', 'score_diff', 'seconds_remaining']) 
    
    df = df[(df['Down'] >= 1) & (df['Down'] <= 4)]
    
    print(f"Filtered {start_len - len(df)} rows.")
    
    df['recommended_cover'] = df.apply(recommend_cover_scheme, axis=1)
    
    final_df = df[[
        'Down', 'Distance', 'YardsToGoal', 
        'offensive_formation', 'play_direction', 
        'play_type_clean', 'recommended_cover',
        'score_diff', 'seconds_remaining', 'team_pass_rate',
        'Offense'
    ]].rename(columns={
        'Down': 'down',
        'Distance': 'distance', 
        'YardsToGoal': 'yard_line',
        'play_type_clean': 'play_type',
        'Offense': 'offense_team'
    })
    
    final_df.to_csv(output_file, index=False)
    print(f"Saved to {output_file}. Rows: {len(final_df)}")

if __name__ == "__main__":
    process_data('download.csv', 'processed_ncaa_features.csv')
