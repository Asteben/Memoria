from csv import DictReader, DictWriter

emergency = 'Earthquake'

with open(f'../Data/preadjusted_data/PreTest{emergency}.csv', 'r', encoding="utf-8") as csv_obj:
    csv = DictReader(x.replace('\0', '') for x in csv_obj)
    csv = sorted(csv, key = lambda row: int(row['created_at'])) 
    rowCount = 0
    Unixinit = 0
    with open(f'../Data/test_data/Test{emergency}.csv', 'w', encoding="utf-8", newline='') as csv_obj_w:
        csvf = DictWriter(csv_obj_w, fieldnames = ['created_at', 'text', 'label'])
        #csvf.writerow(dict((fn,fn) for fn in csv.fieldnames))
        csvf.writeheader()
        for  row in csv:
            rowCount = rowCount + 1
            #if rowCount > 50:
            #   break
            if rowCount == 1:
                Unixinit = int(row['created_at'])
            Unixdif = int(row['created_at']) - Unixinit
            csvf.writerow({'created_at': Unixdif, 'text': row['text'], 'label':row['label']})
            print(f'RowCount:{rowCount}', end = '\r')
    print(f'RowCount:{rowCount}')