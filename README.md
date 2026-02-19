## Website
The website this is hosted at is: https://arlean-monocotyledonous-surreally.ngrok-free.dev
It may go down once in a while as it is hosted on my own computer, so be sure to check a couple times.

## Usage
1.  Click on code, then download zip
2.  Extract the zip file
3.  Click until you open the folder with six files
4.  Click on the path (Next to the Search button in File Explorer)
5.  Type cmd
6.  **Install Requirements**
    ```bash
    pip install -r requirements.txt
    ```

7.  **Run the Web Interface**
    ```bash
    streamlit run app.py
    ```
    or if that doesn't work
    ```bash
    python -m streamlit run app.py
    ```
8.  **Train a Custom Model**
    -   Go to the "Data Manager" tab.
    -   Select an opponent (For example, Army).
    -   Click Train Model.

## Accuracy & Performance
-   **Generic Model**: ~66% Accuracy (With all NCAA teams).
-   **Specialized Model**: Up to 91% Accuracy when trained on teams that have consistent/rigid offenses (For example, Triple Option offenses).

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





