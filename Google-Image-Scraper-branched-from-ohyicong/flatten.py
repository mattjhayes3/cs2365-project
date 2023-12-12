import pandas as pd
from ast import literal_eval
import shutil
import os

df = pd.read_csv('/Users/matthewhayes/Documents/CS236 Deep Generative Models/Project/Google-Image-Scraper/creative and default merged drawings removed.csv')
outdir = 'photos v0.02 flat'
df['alt'] = df['alt'].apply(literal_eval)
if os.path.exists(outdir):
    shutil.rmtree(outdir)
os.makedirs(outdir)
with open(f'{outdir}/metadata.jsonl', 'w') as f:
    for id, row in df.iterrows():
        path = row['path']
        index = path.rindex('.')
        photopath = row['path']
        assert os.path.exists(photopath)
        newphotopath = photopath.replace('/Users/matthewhayes/Documents/CS236 Deep Generative Models/Project/Google-Image-Scraper/creativecommons', 
                        f'/Users/matthewhayes/Documents/CS236 Deep Generative Models/Project/Google-Image-Scraper/{outdir}/creativecommons')
        newphotopath = newphotopath.replace('/Users/matthewhayes/Documents/CS236 Deep Generative Models/Project/Google-Image-Scraper/photos', 
                        f'/Users/matthewhayes/Documents/CS236 Deep Generative Models/Project/Google-Image-Scraper/{outdir}/photos')
        newphotopath = newphotopath.replace('/Users/matthewhayes/Documents/CS236 Deep Generative Models/Project/Google-Image-Scraper/photos2', 
                        f'/Users/matthewhayes/Documents/CS236 Deep Generative Models/Project/Google-Image-Scraper/{outdir}/photos2')
        newphotopath_prefix = f"/Users/matthewhayes/Documents/CS236 Deep Generative Models/Project/Google-Image-Scraper/{outdir}/"
        newphotopath_suffix = newphotopath[len(newphotopath_prefix):]
        newphotopath_suffix = newphotopath_suffix.replace("/", "_")
        newphotopath = newphotopath_prefix + newphotopath_suffix
        dirname = os.path.dirname(newphotopath)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        print(newphotopath_suffix)
        shutil.copy(photopath, newphotopath)
        caption = ', '.join(row['alt']).strip().replace('"', '\\"')
        f.write(f'{{"file_name": "{newphotopath_suffix}", "text": "{caption}"}}\n')