from csv import DictReader, DictWriter
import time, datetime

with open('../../Data/rawdataset/harvey-irma-maria_hydrated.csv', 'r', encoding="utf-8") as csv_obj:
    csv = DictReader(x.replace('\0', '') for x in csv_obj)
    rowCount = 0
    rowCountClean = 0
    with open('../../Data/rawdataset/HurricaneTweetsLimpio.csv', 'w', encoding="utf-8", newline='') as csv_obj_w:
        csvf = DictWriter(csv_obj_w, fieldnames = ['id', 'created_at', 'text', 'label'])
        #csvf.writerow(dict((fn,fn) for fn in csv.fieldnames))
        csvf.writeheader()
        for  row in csv:
            rowCount = rowCount + 1
            #if rowCount > 50:
            #   break
            if "RT" in row['text']:
                aux=0
            else:
                rowCountClean = rowCountClean + 1
                rowsplit = row['created_at'].split()
                rowf = rowsplit[:4]
                rowf.append(rowsplit[5])
                rowjoin = ' '.join(rowf)
                UnixTime = int((datetime.datetime.strptime(rowjoin, "%a %b %d %H:%M:%S %Y")).timestamp())
                csvf.writerow({'id':rowCountClean, 'created_at':UnixTime, 'text': row['text'], 'label':'hurricane'})
            print(f'RowCount:{rowCount}     RowCountClean:{rowCountClean}')