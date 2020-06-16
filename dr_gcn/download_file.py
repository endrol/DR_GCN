# import wget
#
# uri = 'https://drive.google.com/uc?export=download&id=1JYT3_D5jMZpWqB9pdfgaxOHDgmeoFrMz'
# # 'https://drive.google.com/file/d/1cec5BPMKhnkX9TA5FaeYmzJooHFh2_p1/view?usp=sharing'
# # 'https://drive.google.com/file/d/1JYT3_D5jMZpWqB9pdfgaxOHDgmeoFrMz/view?usp=sharing'
#
# wget.download(url=uri, out='/media/kamatalab/78cde73a-a99c-4bcc-b0af-7ba8c7da32f3/dan/Aiki_data/DataSet/data.zip')

import urllib
import requests

url = 'https://drive.google.com/uc?export=download&id=1JYT3_D5jMZpWqB9pdfgaxOHDgmeoFrMz'
print('downloading with urllib')
urllib.request.urlretrieve(url, "/media/kamatalab/78cde73a-a99c-4bcc-b0af-7ba8c7da32f3/dan/Aiki_data/DataSet/data.zip")

# print
# "downloading with urllib2"
# f = urllib2.urlopen(url)
# data = f.read()
# with open("code2.zip", "wb") as code:
#     code.write(data)
#
# print
# "downloading with requests"
# r = requests.get(url)
# with open("code3.zip", "wb") as code:
#     code.write(r.content)
#
