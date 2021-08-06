import pandas as pd

df = pd.read_csv('./test.csv',index_col=0)
#df['image_path'] = './img_align_celeba/' + df['image_path']
print(df.shape)
i = 0
df = df.sort_values(by=["label"], ascending=True)
#df = df.loc[df["label"] >= 300, :]
#df = df.query('0 <= label < 2000')

#train_df = pd.DataFrame(data=None, index=None, columns=None, dtype=None, copy=False)
#test_df = pd.DataFrame(data=None, index=None, columns=None, dtype=None, copy=False)
#val_df =  pd.DataFrame(data=None, index=None, columns=None, dtype=None, copy=False)

for label,sdf in df.groupby('label'):

#exclusive small datasets.
  #if sdf.shape[0] < 30:
  #  continue

  if i == 0:
    sdf.label = i
    val_df = sdf.sample(n=1, random_state=0)
    #sdf = sdf.drop(test_df.index)
    #sdf.drop(test_df.index)
    test_df = sdf.drop(val_df.index)
  elif i !=0:
    sdf.label = i
    sample_df = sdf.sample(n=1, random_state=0)
    val_df = val_df.append(sample_df)
    #sdf.drop(test_df.index,ignore)
    sdf = sdf.drop(sample_df.index)
    test_df = test_df.append(sdf) 
    #sdf.sample(n=2, random_state=0)
  i += 1

print(test_df)
print(val_df)
test_df.to_csv('test.csv')
val_df.to_csv('val.csv')
