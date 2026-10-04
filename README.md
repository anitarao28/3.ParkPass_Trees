# Decision Trees Lab (Streamlit)

An interactive version of the four-stop decision-trees lab on the park-pass data.

| Tab | Business question |
|---|---|
| Stop 1 | Does the bundle work, and is it the same everywhere? (simple logit vs. tree; toggle the interaction term) |
| Stop 2 | Who is worth an offer at all? (the tree's labels vs. expected profit per offer) |
| Stop 3 | Should we pay for age data? (grow the tree; training vs. hidden-customer accuracy) |
| Stop 4 | Who are our most valuable customers? (regression tree for spend; value per offer) |

Business assumptions (profit per pass, bundle cost, cost per offer, visits, share of spend) are in the left sidebar.

## Files
- `app.py`: the app
- `ParkPassSpend.csv`: the data (ParkPass.csv plus the added Age and simulated Spend columns)
- `requirements.txt`: Python packages
- `.streamlit/config.toml`: theme and larger base font for projecting

## Put it online (Streamlit Community Cloud, free)
1. Create a new GitHub repository (it can be private) and upload all the files in this folder, keeping the `.streamlit` folder.
2. Go to https://share.streamlit.io, sign in with GitHub, and click **Create app**.
3. Pick the repository, branch `main`, and main file `app.py`. Click **Deploy**.
4. Share the app's URL with students. Nothing to install on their side.

Notes: free apps go to sleep after a period of no use; the first visitor clicks "wake up" and waits about a minute. Open it yourself a few minutes before class. If the repository is private, set the app's sharing to "Anyone with the link" (or invite students) in the app's settings.

## Run it on your own computer
```
pip install -r requirements.txt
streamlit run app.py
```
