import pandas as pd

fname = '../dataset.csv'
reader = pd.read_csv(fname, chunksize=1000)
#print(type(textreader))
for i,df in enumerate(reader):
  #df = df.drop("unnamed",axis=1)
  print(df)
  df.to_csv('../dataset_csv/dataset' + str(i) + '.csv',index=False)
