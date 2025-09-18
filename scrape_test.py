import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'pinscrape'))
from pinscrape.v2 import Pinterest


keyword = "messi"
output_folder = "output"
images_to_download = 10
number_of_workers = 10

p = Pinterest()
images_url = p.search(keyword, images_to_download)
p.download(url_list=images_url, number_of_workers=number_of_workers, output_folder=output_folder)
