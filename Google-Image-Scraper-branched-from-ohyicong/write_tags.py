import pandas as pd
from ast import literal_eval
import shutil
import os

df = pd.read_csv('/Users/matthewhayes/Documents/CS236 Deep Generative Models/Project/Google-Image-Scraper/creative and default merged drawings removed.csv')
outdir = 'photos v0.02'
df['alt'] = df['alt'].apply(literal_eval)
shutil.rmtree(outdir)
for id, row in df.iterrows():
    path = row['path']
    index = path.rindex('.')
    path = path[:index]+'.caption'
    with open(path, 'w') as f:
        f.write(', '.join(row['alt']))
    photopath = row['path']
    assert os.path.exists(photopath)
    newphotopath = photopath.replace('/Users/matthewhayes/Documents/CS236 Deep Generative Models/Project/Google-Image-Scraper/creativecommons', 
                      f'/Users/matthewhayes/Documents/CS236 Deep Generative Models/Project/Google-Image-Scraper/{outdir}/creativecommons')
    newphotopath = newphotopath.replace('/Users/matthewhayes/Documents/CS236 Deep Generative Models/Project/Google-Image-Scraper/photos', 
                      f'/Users/matthewhayes/Documents/CS236 Deep Generative Models/Project/Google-Image-Scraper/{outdir}/photos')
    newphotopath = newphotopath.replace('/Users/matthewhayes/Documents/CS236 Deep Generative Models/Project/Google-Image-Scraper/photos2', 
                      f'/Users/matthewhayes/Documents/CS236 Deep Generative Models/Project/Google-Image-Scraper/{outdir}/photos2')
    dirname = os.path.dirname(newphotopath)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    shutil.copy(photopath, newphotopath)

    captionpath = path.replace('/Users/matthewhayes/Documents/CS236 Deep Generative Models/Project/Google-Image-Scraper/creativecommons', 
                      f'/Users/matthewhayes/Documents/CS236 Deep Generative Models/Project/Google-Image-Scraper/{outdir}/creativecommons')
    captionpath = captionpath.replace('/Users/matthewhayes/Documents/CS236 Deep Generative Models/Project/Google-Image-Scraper/photos', 
                      f'/Users/matthewhayes/Documents/CS236 Deep Generative Models/Project/Google-Image-Scraper/{outdir}/photos')
    captionpath = captionpath.replace('/Users/matthewhayes/Documents/CS236 Deep Generative Models/Project/Google-Image-Scraper/photos2', 
                      f'/Users/matthewhayes/Documents/CS236 Deep Generative Models/Project/Google-Image-Scraper/{outdir}/photos2')
    shutil.copy(path, captionpath)
    