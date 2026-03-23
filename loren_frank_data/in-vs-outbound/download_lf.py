# %%
import requests
import urllib.parse
import urllib.request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tarfile


host = 'https://datadryad.org'
dataset_doi = 'doi:10.7272/Q61N7ZC3'

url = f'{host}/api/v2/datasets/{urllib.parse.quote(dataset_doi, safe='')}'
response = requests.get(url) 
last_version_url = response.json()['_links']['stash:version']['href']

url = f'{host}{last_version_url}/files'
response = requests.get(url)
files = response.json()

file_name = 'Frank.tar.gz'

file_json = None
for f in files['_embedded']['stash:files']:
    if f['path'] == file_name:
        file_json = f

if file_json == None:
    raise SystemExit('File not found')

download_url = f'{host}{file_json['_links']['stash:download']['href']}'

with tarfile.open(download_url, 'r:gz') as tar:
    for member in tar.getmembers():
        print(member.name)
# %%