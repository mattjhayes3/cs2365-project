import pandas as pd

df = pd.read_csv('/Users/matthewhayes/Documents/CS236 Deep Generative Models/Project/Google-Image-Scraper/creative and default merged.csv')
df = df.loc[~df['query'].str.contains('drawing of')]
df.to_csv('creative and default merged drawings removed.csv')