import urllib
import zipfile
import os

class GraphDataset(object):

    DS_NAMES = {
        'zachary':{
            'url':'http://www-personal.umich.edu/~mejn/netdata/karate.zip',
        }
    }

    def __init__(self, ds_name):
        self.ds_name = ds_name

    def download(self):
        if self.DS_NAMES.get(self.ds_name,None) == None:
            raise Exception('Cannot find datasets')
        else:
            url = self.DS_NAMES[self.ds_name]['url']
            download_file_name = url.split('/').pop()
            data = urllib.request.urlopen(url)
            with open(download_file_name, 'wb') as f:
                f.write(data.read())

            if download_file_name.endswith('.zip'):
                self.__unzip(download_file_name)

    def __unzip(self,file_name):
        with zipfile.ZipFile(file_name) as out_f:
            out_f.extractall()
