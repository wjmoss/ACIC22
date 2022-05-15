import pandas as pd
import sys


## modified from https://blog.csdn.net/qq_29600137/article/details/105307575

def merge(csv_list, outputfile):
    for inputfile in csv_list:
        f = open(inputfile)
        data = pd.read_csv(f, dtype={'level':str, 'year':str})
        data.to_csv(outputfile, mode='a', index=False)
    print('Merging finished')
    
def deduplicate(file, output_file):
    df = pd.read_csv(file, header=None, dtype={'level':str, 'year':str})
    datalist = df.drop_duplicates().fillna('NA')
    datalist.to_csv(output_file, index=False, header=False)
    print('Deduplication finished')
    
if __name__ == '__main__':
    csv_list = ['./res/'+str(500*i+1).zfill(4)+'-'+str(500*i+500).zfill(4)+'.csv' for i in range(6)] + ['./post/3001-3400.csv']
    merge(csv_list, './res/tmp.csv')
    deduplicate('./post/tmp.csv', './res/result.csv')