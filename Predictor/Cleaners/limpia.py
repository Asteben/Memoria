from csv import DictReader, DictWriter

with open('12MCoronaTweets.csv', 'r', encoding="utf-8") as csv_obj:
    csv = DictReader(x.replace('\0', '') for x in csv_obj)
    rowCount = 0
    rowCountClean = 0
    with open('CoronaTweetsLimpio2.csv', 'w', encoding="utf-8", newline='') as csv_obj_w:
        csvf = DictWriter(csv_obj_w, fieldnames = csv.fieldnames)
        csvf.writerow(dict((fn,fn) for fn in csv.fieldnames))
        for  row in csv:
            #if rowCount == 50:
            #   break
            if "RT" in row['text']:
                aux=0
            else:
                csvf.writerow(row)
                rowCountClean = rowCountClean + 1
            rowCount = rowCount + 1
            print(f'RowCount:{rowCount}     RowCountClean:{rowCountClean}')