import json
import csv

with open('../dataset/dataset2.jsonl', 'r',  encoding="utf-8") as data1:
    json_list = list(data1)
    rowCount = 1
    with open('CoronaTweetsLimpio2.csv', 'w', encoding="utf-8", newline='') as csv_obj_w:
        csvf = csv.writer(csv_obj_w)
        for json_str in json_list:
            line = json.loads(json_str)
            line["text"] = line["text"].replace('\n', ' ')
            if rowCount == 1:
                header = line.keys()
                csvf.writerow(header)
            csvf.writerow(line.values())
            #if rowCount == 5:
            #    break
            print(f'RowCount:{rowCount}')
            rowCount = rowCount + 1