# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 11:02:06 2020

@author: OHyic

"""
#Import libraries
import os
from datetime import datetime
import concurrent.futures
from GoogleImageScraper import GoogleImageScraper
from patch import webdriver_executable
import pandas as pd
from threading import Lock

class Results:
    def __init__(self):
        self.df = pd.DataFrame({"path":[], "query":[], "alt":[], "src":[]})
        self.lock = Lock()

    def add(self, path, query, alt, src):
        with self.lock:
            self.df.loc[len(self.df.index)] = [path, query, alt, src]

RESULTS = Results()

def worker_thread(search_key):
    image_scraper = GoogleImageScraper(
        webdriver_path, 
        image_path, 
        search_key, 
        number_of_images, 
        headless, 
        min_resolution, 
        max_resolution, 
        max_missed,
        RESULTS)
    print("built scraper")
    image_urls, alts = image_scraper.find_image_urls()
    image_scraper.save_images(image_urls, keep_filenames, alts)

    #Release resources
    del image_scraper

if __name__ == "__main__":
    #Define file path
    webdriver_path = os.path.normpath(os.path.join(os.getcwd(), 'webdriver', webdriver_executable()))
    image_path = os.path.normpath(os.path.join(os.getcwd(), 'creativecommons'))

    #Add new search key into array ["cat","t-shirt","apple","orange","pear","fish"]
    search_keys = list(set([
                            "hand",
                            "hands",
                            "thumbs up",
                            # "thumbs down", 
                            'handshake', 
                            'palms out', 
                            'fist', 
                            'one finger', 
                            'two fingers', 
                            'three fingers',
                            'four fingers',
                            'five fingers',
                            "okay hand gesture",
                            "backoning hand",
                            "holding hands",
                            "fingers crossed",
                            "horn sign",
                            "waving",
                            "hand heart",
                            "air quotes",
                            "jazz hands",
                            "prayer hands",
                            "clasped hands",
                            "fist"
                            ]))
    # search_keys += [
    #     # 'man showing hand',
    #     # 'photo of fist',
    #     # 'photo of five fingers',
    #     'photo of hand',
    #     # 'photo of hands',
    #     # 'photo of handshake',
    #     # 'photo of one finger',
    #     # 'photo of two fingers',
    #     # 'photo of three fingers',
    #     # 'photo of palms out',
    #     # 'photo of thumbs down',
    #     # 'photo of thumbs up',
    #     # 'photo of woman showing three fingers',
    # ]
    search_keys_raw = list(search_keys)
    search_keys += [
        f"photo of {key}"
        for key in search_keys_raw
    ]
    search_keys += [
        f"drawing of {key}"
        for key in search_keys_raw
    ]
    man_woman = [
        "showing hand",
        "showing hands",
        "giving thumbs up",
        "giving thumbs down", 
        'giving handshake', 
        'holding palms out', 
        'holding fist', 
        'showing one finger', 
        'showing two fingers', 
        'showing three fingers',
        'showing four fingers',
        'showing five fingers',
        "giving okay gesture",
        "beckoning with hand",
        "with fingers crossed",
        "giving horn sign",
        "waving",
        "with hand heart",
        "giving air quotes",
        "with jazz hands",
        "with prayer hands",
        "with clasped hands",
    ]
    search_keys += [
        f"man {key}" for key in man_woman
    ]
    search_keys += [
        f"woman {key}" for key in man_woman
    ]
    search_keys += [
        f"photo of man {key}" for key in man_woman
    ]
    search_keys += [
        f"photo of woman {key}" for key in man_woman
    ]
    search_keys += [
        f"drawing of man {key}" for key in man_woman
    ]
    search_keys += [
        f"drawing of woman {key}" for key in man_woman
    ]
    print(f"all search keys ({len(search_keys)}) : {search_keys}")

    #Parameters
    number_of_images = 10                # Desired number of images
    headless = True                     # True = No Chrome GUI
    min_resolution = (512, 512)             # Minimum desired image resolution
    max_resolution = (9999, 9999)       # Maximum desired image resolution
    max_missed = 10                     # Max number of failed images before exit
    number_of_workers = 6               # Number of "workers" used
    keep_filenames = False              # Keep original URL image filenames

    #Run each search_key in a separate thread
    #Automatically waits for all threads to finish
    #Removes duplicate strings from search_keys
    with concurrent.futures.ThreadPoolExecutor(max_workers=number_of_workers) as executor:
        executor.map(worker_thread, search_keys)
    
    RESULTS.df.to_csv(f'./results-{datetime.now().isoformat()}.csv')
