import pandas as pd

df = pd.read_csv('coinbase-1h-btc-usd.csv')
df = df[['Open','High','Low','Close','VolumeFrom']]

df.rename(columns={'Open': 'open',
                   'High': 'high',
                   'Low' : 'low',
                   'Close':'close',
                   'VolumeFrom':'volumn'
                    }, inplace = True)

if __name__ == '__main__':
    print(df.head())