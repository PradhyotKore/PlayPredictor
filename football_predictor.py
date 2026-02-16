import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib

def load_data(filepath):
    return pd.read_csv(filepath)

def save_models(models, filename='football_model.pkl'):
    joblib.dump(models, filename)
    print(f"Models saved to {filename}")

def load_saved_models(filename='football_model.pkl'):
    return joblib.load(filename)

def train_models(df, target_team=None):
    print("Training models...")
    
    if target_team and target_team in df['offense_team'].unique():
        print(f"\n!!! SPECIALIZED TRAINING: Training EXCLUSIVELY for {target_team} !!!")
        team_df = df[df['offense_team'] == target_team]
        if len(team_df) > 50:
            df = team_df
            print(f"Using {len(df)} plays from {target_team}.")
        else:
            print(f"Not enough data for {target_team} ({len(team_df)} rows). Using generic model.")
    elif target_team:
        print(f"Warning: Team {target_team} not found. Training generic model.")

    feature_cols = ['down', 'distance', 'yard_line', 'score_diff', 'seconds_remaining', 'team_pass_rate', 'offensive_formation']
    
    categorical_features = ['offensive_formation']
    numeric_features = ['down', 'distance', 'yard_line', 'score_diff', 'seconds_remaining', 'team_pass_rate']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='median'), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    models = {}
    scores = {}
    
    print("\n--- Training Play Type Model ---")
    
    X = df[feature_cols]
    y = df['play_type']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    clf_type = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=42))])
    
    clf_type.fit(X_train, y_train)
    y_pred = clf_type.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    print(f"Play Type Accuracy: {score:.2f}")
    print(classification_report(y_test, y_pred))
    
    models['play_type'] = clf_type
    scores['play_type'] = score

    print("\n--- Training Cover Scheme Model ---")
    
    y_cover = df['recommended_cover']
    X_train, X_test, y_train, y_test = train_test_split(X, y_cover, test_size=0.2, random_state=42)
    
    clf_cover = Pipeline(steps=[('preprocessor', preprocessor),
                                ('classifier', RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=42))])
    clf_cover.fit(X_train, y_train)
    y_pred = clf_cover.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    print(f"Cover Scheme Accuracy: {score:.2f}")

    models['cover_scheme'] = clf_cover
    scores['cover_scheme'] = score

    print("\n--- Training Play Direction Model ---")
    
    df_dir = df[df['play_direction'] != 'Unknown']
    
    if len(df_dir) > 50:
        X_dir = df_dir[feature_cols]
        y_dir = df_dir['play_direction']
        
        X_train, X_test, y_train, y_test = train_test_split(X_dir, y_dir, test_size=0.2, random_state=42)
        
        clf_dir = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('classifier', RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=42))])
        clf_dir.fit(X_train, y_train)
        y_pred = clf_dir.predict(X_test)
        score = accuracy_score(y_test, y_pred)
        print(f"Play Direction Accuracy: {score:.2f} (on {len(df_dir)} rows)")
        models['play_direction'] = clf_dir
        scores['play_direction'] = score
    else:
        print(f"Not enough data for Play Direction model. Skipping.")
        models['play_direction'] = None
        scores['play_direction'] = 0.0

    if target_team:
        safe_name = "".join([c for c in target_team if c.isalnum() or c==' ']).rstrip()
        filename = f'football_model_{safe_name}.pkl'
        save_models(models, filename)
    else:
        save_models(models, 'football_model.pkl')
        
    return models, scores

def interactive_mode(models):
    print("\n" + "="*50)
    print("FOOTBALL PLAY PREDICTOR AI")
    print("="*50)
    print("Enter game situation to get predictions.")
    
    while True:
        try:
            print("\nType 'exit' to quit.")
            down = input("Down (1-4): ")
            if down.lower() == 'exit': break
            down = int(down)
            
            distance = float(input("Distance (1-99): "))
            yard_line = float(input("Yard Line (1-99 yards to goal): "))
            formation = input("Offensive Formation (Shotgun, Pistol, Trips, Empty, Standard): ")
            score_diff = float(input("Score Differential (Offense - Defense, e.g. -7): "))
            seconds = float(input("Seconds Remaining (e.g. 900 for 15 mins): "))
            pass_rate = float(input("Team Hist. Pass Rate (0.0-1.0, e.g. 0.80 for Air Raid): "))
            
            input_data = pd.DataFrame({
                'down': [down],
                'distance': [distance],
                'yard_line': [yard_line],
                'score_diff': [score_diff],
                'seconds_remaining': [seconds],
                'team_pass_rate': [pass_rate],
                'offensive_formation': [formation]
            })
            
            print("\n--- PREDICTION ---")
            
            type_probs = models['play_type'].predict_proba(input_data)[0]
            type_classes = models['play_type'].classes_
            print(f"Play Type: {models['play_type'].predict(input_data)[0]}")
            for c, p in zip(type_classes, type_probs):
                print(f"  {c}: {p*100:.1f}%")
                
            print(f"Recommended Cover: {models['cover_scheme'].predict(input_data)[0]}")
            
            if models['play_direction']:
                dir_probs = models['play_direction'].predict_proba(input_data)[0]
                dir_classes = models['play_direction'].classes_
                print(f"Play Direction: {models['play_direction'].predict(input_data)[0]}")
                for c, p in zip(dir_classes, dir_probs):
                    print(f"  {c}: {p*100:.1f}%")
            else:
                print("Play Direction: (Insufficient training data)")
            
        except ValueError:
            print("Invalid input. Please enter numbers.")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default='processed_ncaa_features.csv', help='Path to processed data')
    parser.add_argument('--test-only', action='store_true', help='Run training and exit')
    parser.add_argument('--target_team', type=str, default=None, help='Name of team to oversample for custom model')
    args = parser.parse_args()
    
    try:
        df = load_data(args.file)
        models, scores = train_models(df, args.target_team)
        
        if not args.test_only:
            interactive_mode(models)
    except FileNotFoundError:
        print(f"Error: File {args.file} not found. Please run ncaa_data_loader.py first.")
