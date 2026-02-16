# Football AI Predictor

An AI-powered tool that predicts offensive play calls in real-time using Machine Learning. Designed to help High School coaches make decision based off of data on the sideline.

## The Mission
At the high school level, 85% of play-calling is based on gut instinct. This project aims to give analytics to underfunded teams by providing NFL-level insights using a simple, offline program.

## How It Works
The system uses **Random Forest Classifiers** trained on thousands of NCAA play-by-play data points to predict:
1.  **Play Type** (Run vs. Pass)
2.  **Cover Scheme** (Defensive Recommendation)
3.  **Play Direction** (Left/Right/Middle)

### Key Features
-   **Real-Time Prediction**: Enter Down, Distance, and Formation to get instant probabilities.
-   **Opponent Specific**: The AI can be trained on a specific opponent's history (For example, Army) to learn their unique tendencies (The accuracy relies on the amount of data you have on the team)
-   **Defensive Coverage**: Suggests the "best" defensive coverage based on expected play.

## Usage
1.  **Install Requirements**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the Web Interface**
    ```bash
    streamlit run app.py
    ```
    or if that doesn't work
    ```bash
    python -m streamlit run app.py
    ```
4.  **Train a Custom Model**
    -   Go to the "Data Manager" tab.
    -   Select an opponent (For example, Army).
    -   Click Train Model.

## Accuracy & Performance
-   **Generic Model**: ~66% Accuracy (With all NCAA teams).
-   **Specialized Model**: Up to **91% Accuracy** when trained on teams that have consistent/rigid offenses (For example, Triple Option offenses).

## Technology Used
-   **Python 3.10+**
-   **Scikit-Learn** (Machine Learning)
-   **Pandas** (Data Processing)
-   **Streamlit** (UI)
## Extras
-   The large dataset I used was too big to upload to GitHub
-   This was made for own school's football team
-   All data came from https://collegefootballdata.com/

---
*Created by Pradhyot Kore and Kaushal Rao



