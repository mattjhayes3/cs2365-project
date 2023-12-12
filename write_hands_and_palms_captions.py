import pandas as pd
import random

df = pd.read_csv('hands and palms/HandInfo.csv')
df = df.loc[df['irregularities'] == 0]
df['front'] = df['aspectOfHand'].str.contains('palmar')
df['left'] = df['aspectOfHand'].str.contains('left')

def make_caption(row):
    side = "front" if row['front'] else 'back'
    lr = "left" if row['left'] else 'right'
    extras = ""
    if row['accessories']==1 and row['nailPolish']==1:
        extras = " with accessories and nail polish" if random.randint(0,1) == 0 else " with nail polish and accessories"
    elif row['accessories']==1:
        extras = " with accessories"
    elif row['nailPolish']==1:
        extras = " with nail polish"
    age = "" if row['age'] < 60 else " elderly"
    return f"{side} of {row['skinColor']}-skinned{age} {row['gender']} {lr} hand{extras}"

df['text'] = df.apply(make_caption, axis=1)
df['file_name'] = df['imageName']
df[['file_name','text']].to_csv("hands and palms/metadata.csv", index=False)