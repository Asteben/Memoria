from csv import DictReader, DictWriter

emergency = 'Earthquake'

with open(f'../Data/raw_data/{emergency}TweetsClean.csv', 'r', encoding="utf-8") as csv:
    csv = DictReader(x.replace('\0', '') for x in csv)
    csv = sorted(csv, key = lambda row: row['created_at'])
    rowTotal = sum(1 for row in csv) 
    print(f'RowTotal:{rowTotal}')
    rowCount = 0
    Unixinit = 0
    Aux = 0
    with open(f'../Data/train_data/Train{emergency}.csv', 'w', encoding="utf-8", newline='') as csv_train:
        csv_trn = DictWriter(csv_train, fieldnames = ['created_at', 'text', 'label'])
        csv_trn.writeheader()
        with open(f'../Data/preadjusted_data/PreTest{emergency}.csv', 'w', encoding="utf-8", newline='') as csv_test_c:
            csv_tst_c = DictWriter(csv_test_c, fieldnames = ['created_at', 'text', 'label'])
            csv_tst_c.writeheader()
            for  row in csv:
                rowCount = rowCount + 1
                if rowCount < rowTotal*0.8:
                    csv_trn.writerow({'created_at': int(row['created_at']), 'text': row['text'], 'label':row['label']})
                else:
                    csv_tst_c.writerow({'created_at': int(row['created_at']), 'text': row['text'], 'label':row['label']})

                print(f'RowCount:{rowCount}', end = '\r')
    print(f'RowCount:{rowCount}')