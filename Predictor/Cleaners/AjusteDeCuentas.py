from csv import DictReader, DictWriter

with open('../../Data/rawdataset/CoronaTweetsLimpio4.csv', 'r', encoding="utf-8") as csv_obj:
    csv = DictReader(x.replace('\0', '') for x in csv_obj)
    csv = sorted(csv, key = lambda row: row['created_at']) 
    rowCount = 0
    Unixinit = 0
    with open('../../Data/rawdataset/Corona_Ajustado.csv', 'w', encoding="utf-8", newline='') as csv_obj_w:
        csvf = DictWriter(csv_obj_w, fieldnames = ['created_at', 'text', 'label'])
        #csvf.writerow(dict((fn,fn) for fn in csv.fieldnames))
        csvf.writeheader()
        for  row in csv:
            rowCount = rowCount + 1
            #if rowCount > 50:
            #   break
            if rowCount == 1:
                Unixinit = row['created_at']
            Unixdif = row['created_at'] - Unixinit
            csvf.writerow({'created_at': Unixdif, 'text': row['text'], 'label':row['label']})
            print(f'RowCount:{rowCount}')