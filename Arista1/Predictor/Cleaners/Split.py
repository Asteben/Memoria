from csv import DictReader, DictWriter

with open('../../Data/rawdataset/HurricaneTweetsLimpio.csv', 'r', encoding="utf-8") as csv:
    csv = DictReader(x.replace('\0', '') for x in csv)
    csv = sorted(csv, key = lambda row: row['created_at'])
    rowTotal = sum(1 for row in csv) 
    print(f'RowTotal:{rowTotal}')
    rowCount = 0
    Unixinit = 0
    Aux = 0
    with open('../../FinalSystem/Data/TrainHurricane.csv', 'w', encoding="utf-8", newline='') as csv_train:
        csv_trn = DictWriter(csv_train, fieldnames = ['created_at', 'text', 'label'])
        csv_trn.writeheader()
        with open('../../FinalSystem/Data/TestHurricane-C.csv', 'w', encoding="utf-8", newline='') as csv_test_c:
            csv_tst_c = DictWriter(csv_test_c, fieldnames = ['created_at', 'text', 'label'])
            for  row in csv:
                rowCount = rowCount + 1
                if rowCount < rowTotal*0.6:
                    csv_trn.writerow({'created_at': int(row['created_at']), 'text': row['text'], 'label':row['label']})
                else:
                    if rowCount < rowTotal*0.8:
                        if Aux == 0:
                            Unixinit = int(row['created_at'])
                            Aux = 1
                        Unixdif = int(row['created_at']) - Unixinit
                        csv_tst_c.writerow({'created_at': Unixdif, 'text': row['text'], 'label':row['label']})

                print(f'RowCount:{rowCount}', end = '\r')