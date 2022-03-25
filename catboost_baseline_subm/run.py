import sys
import pickle
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
import gc

def main():
    data, output_path = sys.argv[1:]
    transactions = pd.read_csv(f'{data}/transactions.csv')
    transactions['transaction_dttm'] = pd.to_datetime(transactions['transaction_dttm'])
    transactions_dtypes = {'mcc_code':np.int16, 'currency_rk':np.int8}
    transactions = transactions.astype(transactions_dtypes)
    bankclient_embed = transactions .pivot_table(index = 'user_id', 
                            values=['transaction_amt'],
                            columns=['mcc_code'],
                            aggfunc=['sum','mean', 'count']).fillna(0)
    bankclient_embed.columns = [f'bank_{str(i[0])}-{str(i[2])}' for i in bankclient_embed.columns]
    
    del transactions
    gc.collect()
    
    clickstream = pd.read_csv(f'{data}/clickstream.csv')
    clickstream['timestamp'] = pd.to_datetime(clickstream['timestamp'])
    clickstream_dtypes = {'cat_id':np.int16, 'new_uid':np.int32}
    clickstream = clickstream.astype(clickstream_dtypes)
    clickstream_embed = clickstream.pivot_table(index = 'user_id', 
                            values=['timestamp'],
                            columns=['cat_id'],
                            aggfunc=['count']).fillna(0)
    clickstream_embed.columns = [f'rtk_{str(i[0])}-{str(i[2])}' for i in clickstream_embed.columns]
    clickstream_embed.loc[0] = np.empty(len(clickstream_embed.columns))

    del clickstream
    gc.collect()

    dtype_clickstream = list()
    for x in clickstream_embed.dtypes.tolist():
        if x=='int64':
            dtype_clickstream.append('int16')
        elif(x=='float64'):
            dtype_clickstream.append('float32')
        else:
            dtype_clickstream.append('object')

    dtype_clickstream = dict(zip(clickstream_embed.columns.tolist(),dtype_clickstream))
    clickstream_embed = clickstream_embed.astype(dtype_clickstream)

    dtype_bankclient = list()
    for x in bankclient_embed.dtypes.tolist():
        if x=='int64':
            dtype_bankclient.append('int16')
        elif(x=='float64'):
            dtype_bankclient.append('float32')
        else:
            dtype_bankclient.append('object')
    
    dtype_bankclient = dict(zip(bankclient_embed.columns.tolist(),dtype_bankclient))
    bankclient_embed = bankclient_embed.astype(dtype_bankclient)

    list_of_rtk = list(clickstream_embed.index.unique())
    list_of_bank= list(bankclient_embed.index.unique())
    
    submission = pd.DataFrame(list_of_bank, columns=['bank'])
    submission['rtk'] = submission['bank'].apply(lambda x: list_of_rtk)

    with open("full_list_of_features_baseline", "rb") as fp:   # Unpickling
        full_list_of_features = pickle.load(fp)

    model = CatBoostClassifier()
    model.load_model('model_baseline.cbm',  format='cbm')

    submission_ready = []

    batch_size = 200
    num_of_batches = int((len(list_of_bank))/batch_size)+1

    for i in range(num_of_batches):
        bank_ids = list_of_bank[(i*batch_size):((i+1)*batch_size)]
        if len(bank_ids) != 0:
            part_of_submit = submission[submission['bank'].isin(bank_ids)].explode('rtk')
            part_of_submit = part_of_submit.merge(bankclient_embed, how='left', left_on='bank', right_index=True
                                        ).merge(clickstream_embed, how='left', left_on='rtk', right_index=True).fillna(0)
        
            for i in full_list_of_features:
                if i not in part_of_submit.columns:
                    part_of_submit[i] = 0
            

            part_of_submit['predicts'] = model.predict_proba(part_of_submit[full_list_of_features])[:,1]
            part_of_submit = part_of_submit[['bank', 'rtk', 'predicts']]

            zeros_part = pd.DataFrame(bank_ids, columns=['bank'])
            zeros_part['rtk'] = 0.
            zeros_part['predicts'] = 3.8
            
            part_of_submit = pd.concat((part_of_submit, zeros_part))

            part_of_submit = part_of_submit.sort_values(by=['bank', 'predicts'], ascending=False).reset_index(drop=True)
            part_of_submit = part_of_submit.pivot_table(index='bank', values='rtk', aggfunc=list)
            part_of_submit['rtk'] = part_of_submit['rtk'].apply(lambda x: x[:100])
            part_of_submit['bank'] = part_of_submit.index
            part_of_submit = part_of_submit[['bank', 'rtk']]
            submission_ready.extend(part_of_submit.values)
    
    submission_final = np.array(submission_ready, dtype=object)

    print(submission_final.shape)
    np.savez(output_path, submission_final)

if __name__ == "__main__":
    main()