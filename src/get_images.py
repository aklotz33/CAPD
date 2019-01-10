import os
from collections import defaultdict
import time
import requests
import re

from unidecode import unidecode
from bs4 import BeautifulSoup

from util import REPO_STR


def progress_bar(current, total, label='', length=40):
    bar = '[{}{}] '.format('#'*int(length*current/total), '-'*(length - int(length*current/total))) \
        + '{:.2f}% '.format(100*current/total) + label
    print(bar, end='\x1b[K\r')  # \x1b[K erases to the end of the line and \r returns you to the start

# Need to manually download raw_html.html from https://www.politicsanddesign.com/ by scrolling to the bottom.
# Could replace with Selenium to get the html with Python
with open(REPO_STR+'/data/pretty_html.html') as f:
    bs = BeautifulSoup(f, features='lxml')

# Find all of the "Campaign" images
people = bs.find_all('img', attrs={'src': True, 'alt': re.compile(r'Campaign')})

# Create dict of name: image link
links = defaultdict()
for name in people:
    # Extract the name from "Campaign logo for A. Persons Name", removes '.', and replaces spaces with '_'
    key = name['alt'].replace('Campaign logo for ', '').replace(' ', '_').replace('.', '')
    links[key] = name['src'].split(' ')[0]

# Make /images directory to store images
if not os.path.exists(REPO_STR+'/images'):
    os.mkdir(REPO_STR+'/images')
# Loop through all the names and link suffixes to request images and save to images/name.jpg
base_url = 'https://www.politicsanddesign.com'
for i, (k, v) in enumerate(links.items()):
    progress_bar(i, len(links), label=k)
    while True:
        r = requests.get(''.join([base_url, '/', v]))
        if r.status_code == 200:
            with open(''.join([REPO_STR, '/images/', unidecode(k), '.jpg']), 'wb') as f:
                f.write(r.content)
            break
        else:
            time.sleep(61)  # if the request code is not 200, wait 1 min and try again

print('Done!')
