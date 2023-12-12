# Financial Market Analysis Project

## Introduction
This project involves the analysis of financial market data, focusing on major US market indices since 2013. The goal is to explore various data analysis techniques using Python and create interactive visualizations to understand market trends.

## Technologies Used
- Python
- Pandas
- Plotly
- Dash

## Features
- Data preprocessing and cleanup to handle non-standard data formats.
- Calculation and visualization of rolling averages for market indices.
- Analysis of market behavior around major events like the Covid-19 pandemic.
- Development of an interactive dashboard for data exploration.

## How to Run
1. Clone the repository: `git clone [repository-url]`
2. Install required Python packages: `pip install -r requirements.txt`
3. Run the application: `python dashboard.py`

## Challenges and Learning
- Ran into some challenges and errors involving the periods in the PRICE columns of data. Needed to make sure to account for additional data cleaning. 
- For all NaN data cells, I elected to forward the PRICE from the day previous to account for days the market was closed.
- Some of my initial attempts at visual graphs resulted in straight diagonal lines from bottom left to top right. Changing to rolling averages helped make visuals more realistic to actual market flow.

## Conclusion
- Learned a lot about analysis and python. ChatGPT4 was amazing as my troubleshooting partner. I see a lot of programmers electing to use additional AI tools to help code more efficiently. Even with my novice skills, I was able to put together 70+ lines of code quickly with minimal errors.