import pandas as pd
import os

current = '/Users/matthewhayes/Documents/CS236 Deep Generative Models/Project/Google-Image-Scraper/results-2023-11-11T18:37:54.140080_deduped_edited_merged.csv'
current = pd.read_csv(current)
new = '/Users/matthewhayes/Documents/CS236 Deep Generative Models/Project/Google-Image-Scraper/results-2023-11-11T22:16:18.759809.csv'


df = pd.read_csv(new)

# print(df['src'])
# print(current['src'])

def delDupes(paths):
    for path in paths[1:]:
        os.remove(path)
        print('del', path)
    return paths[0]

df = df.loc[df.path.apply(lambda p: os.path.exists(p))]
df = df.groupby('src').agg(list)
df['path'] = df['path'].apply(delDupes)

incurrent = df.index.isin(current['src'])
for index, row in df.loc[incurrent].iterrows():
    os.remove(row['path'])
    print('del', row['path'])
df = df.loc[~incurrent]

df.to_csv(f"{new[:-4]}_deduped.csv")
