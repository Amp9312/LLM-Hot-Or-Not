#Licensed under the Apache License, Version 2.0
# Last tested on Python 3.13.11
#Goal: Query Github API for summary statistics of top % of stars/day between general and LLM related repos
#Secondary Goal: Build Linear Regression Model that predicts trajectory of LLM repos given current data

#What is breakout success?
#A phenomenae in coding referring to a sudden overwhelming achievement/rapid growth of a developer product (in this analysis, LLMs)
#For simplicity of analysis, the star system will be utilized to see how a certain category of project is performing in the social setting of the GitHub developer community
# Categories:
# AI - LLMs
# DevTools
# Web Frameworks
# Data Science Tools


#Interesting observations
# 1. Can't include year 2026 in scope as it may include a dataset that is too small to analyze (current month if month just started)
# 2. Can't use t-test bc it only checks for rate of growth, I need to see if these categories are indendenpent
# 3. Comparing against randomly sampled general github repos is too noisy to compare against LLMs, causing bias in the model (pivoting to using multiple areas)
# 4. Analyzing one random month may not prove replicable, as months may have variance in activity. Stablize results by randomly sampling instead

#Dependencies
# NOTE: Need to generate own Github Personal Access Token and add it as an environment variable in order for script to work. Sorry, not sharing mine!
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import time
import requests
import os
import numpy as np

#for time slicing / random sampling
import random

#set up API access
url = "https://api.github.com/search/repositories"

HEADERS = {
    "Authorization" : f"token {os.getenv('GITHUB_TOKEN')}",
    "Accept" : "application/vnd.github+json"
}

#verify its using the access token
response = requests.get("https://api.github.com/rate_limit",headers=HEADERS)
print(response.json())
print("Token loaded: ", os.getenv('GITHUB_TOKEN'))

#gen random slice of time window (maybe come back and modify this to only capture certain years/dates. TODO: Make evergreen)
def random_date_range():
    year=random.randint(2022, 2025)
    month=random.randint(1,12)

    start=f"{year}-{month:02d}-01"
    end=f"{year}-{month:02d}-28"
    return start, end

start, end=random_date_range()

#build mega query
def build_queries(start, end):
    return {
        "LLM": f"(llm OR gpt OR transformer OR 'language model') created:{start}..{end} stars:>1",

        "DevTools": f"(cli OR 'developer tool' OR devtool OR productivity) created:{start}..{end} stars:>1",

        "Web": f"(react OR vue OR angular OR 'web framework' OR frontend) created:{start}..{end} stars:>1",

        "DataScience": f"(data science OR pandas OR numpy OR visualization) created:{start}..{end} stars:>1"
    }

#extract features for dataset
def fetch_repos(query, max_pages=5):
    repo_data=[]

    for page in range(1, max_pages+1):
        params = {
            "q":query,
            "per_page":100,
            "page":page
        }

        response = requests.get(url, headers=HEADERS, params=params)

        if response.status_code !=200:
            print("Error: ", response.json())
            break

        data = response.json()
        repos = data.get("items", [])

        if not repos:
            break

        for repo in repos:
            repo_data.append({
                "name":repo["name"],
                "stars":repo["stargazers_count"],
                "forks":repo["forks_count"],
                "watchers":repo["watchers_count"],
                "open_issues":repo["open_issues_count"],
                "created_at":repo["created_at"],
                "updated_at":repo["updated_at"],
                "language":repo["language"]
            })

        print(f"Fetched page {page} ({len(repos)} repos)")
        time.sleep(1)

        print("Query:",query)
        print("Total repos fetched so far:", len(repo_data))
    return pd.DataFrame(repo_data)

def clean_and_analyze(df):
    if df.empty:
        raise ValueError("DataFrame is empty - ensure data collection is succeeding!")

    required_cols=["created_at", "updated_at", "stars", "forks"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    # convert to UTC bc timezone naive can't be compared to timezone aware (github api also uses UTC)
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    df["updated_at"] = pd.to_datetime(df["updated_at"], utc=True)

    #format age
    df["repo_age_days"]=(pd.Timestamp.now(tz="UTC") - df["created_at"]).dt.days

    #screen for impossible expressions
    df = df[df["repo_age_days"]>0]

    #defining breakout by finding number of stars per day
    df["stars_per_day"]=df["stars"]/df["repo_age_days"]

    #in case you forget why the .replace is there its also to prevent impossible expressions such as div by 0
    df["fork_ratio"]=df["forks"]/df["stars"].replace(0,1)

    return df

# Run code, grab data (NEW)
all_dfs=[]

for i in range(5): #change this number to change number of samples!!
    print(f"\n--- Sampling iteration {i+1} ---")

    start, end = random_date_range()
    queries = build_queries(start, end)

    for category, query in queries.items():
        print(f"Fetching {category} for {start} to {end}")

        df = fetch_repos(query, max_pages=2)

        if df.empty:
            print("Skipped empty dataset")
            continue

        df = clean_and_analyze(df)
        df["type"] = category

        all_dfs.append(df)

combined_df = pd.concat(all_dfs, ignore_index=True)

#define breakout threshold
threshold = combined_df["stars_per_day"].quantile(0.90)

#label breakout
combined_df["breakout"] = combined_df["stars_per_day"]>=threshold

#compute probability of breakout
breakout_rates = combined_df.groupby("type")["breakout"].mean()
print("\nBreakout Probability by Category:")
print(breakout_rates)

#chi-square test to see if there is stat significance in breakout difference
contingency = pd.crosstab(combined_df["type"], combined_df["breakout"])
chi2, p, _, _ = chi2_contingency(contingency)
print("p-value:",p)

#print stats about data frames
for category in combined_df["type"].unique():
    print(f"\n{category} Summary Stats:")
    print(combined_df[combined_df["type"]==category].describe())

#visualize Probability bar chart
breakout_rates.plot(kind="bar")
plt.title("Breakout Probability by Category")
plt.ylabel("Proportion")
plt.show()

#left some of this in as prebaked examples. Matplotlib has a ton of other examples though!

#visualize star distribution (not effective)
#plt.hist(df_llm["stars"],bins=30,alpha=0.5,label="LLM")
#plt.hist(df_general["stars"],bins=30,alpha=0.5,label="General")
#plt.legend()
#plt.title("Star Distribution")
#plt.show()

#Visualize stars per day as a boxplot to see general distribution
#plt.boxplot([df_llm["stars_per_day"],df_general["stars_per_day"]],labels=["LLM","General"])
#plt.title("Stars per Day Comparison")
#plt.show()

#Visualize trend analysis over time
#df_llm["year_month"]=df_llm["created_at"].dt.to_period("M")
#trend=df_llm.groupby("year_month")["stars"].mean()
#trend.plot()
#plt.title("Average Stars Over Time(LLM)")
#plt.show()

#Save to CSV
combined_df.to_csv("repos.csv",index=False)
print("\nData saved to CSV files.")

########################################
# LINEAR REGRESSION TIME
# ######################################
# note: Data is skewed (tail heavy), will need to adjust for this

df_llm=combined_df[combined_df["type"]=="LLM"].copy()
df_llm["log_stars_per_day"]=np.log1p(df_llm["stars_per_day"])
df_llm["log_forks"] = np.log1p(df_llm["forks"])
df_llm["log_issues"] = np.log1p(df_llm["open_issues"])
df_llm["year"] = pd.to_datetime(df_llm["created_at"]).dt.year

# define features (x) and target (y)

x = df_llm[["log_forks","log_issues", "repo_age_days", "year"]]
y = df_llm["log_stars_per_day"]

# convert to numpy
X_mat = x.values
y_vec = y.values

X_mat = np.column_stack([np.ones(len(X_mat)), X_mat])
beta = np.linalg.lstsq(X_mat, y_vec, rcond=None)[0]

print("\nRegression Coefficients:")
print("Intercept:", beta[0])
print("log_forks:", beta[1])
print("log_issues:", beta[2])
print("repo_age_days:", beta[3])
print("year:", beta[4])

#evaluate
y_pred = X_mat @ beta
ss_res = np.sum((y_vec - y_pred) ** 2)
ss_tot = np.sum((y_vec - np.mean(y_vec)) ** 2)

r2 = 1 - (ss_res / ss_tot)
print("\nR^2 Score: ", r2)

#predict
sample = np.array([1, np.log1p(500), np.log1p(50), 200, 2024])
pred_log = sample @ beta

pred_growth = np.expm1(pred_log)

print("\nPredicted stars/day", pred_growth)

#visualize
plt.scatter(y_vec, y_pred)
plt.xlabel("Actual log(stars_per_day)")
plt.ylabel("Predicted log(stars_per_day)")
plt.title("Predicted vs Actual Growth Rate")

# Perfect prediction line
plt.plot([min(y_vec), max(y_vec)], [min(y_vec), max(y_vec)])

plt.show()
