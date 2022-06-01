from csv import DictReader, DictWriter
import json

with open('CoronaTweetsLimpio2.csv', 'r', encoding="utf-8") as csv_obj:
    csv = DictReader(x.replace('\0', '') for x in csv_obj)
    rowCount = 1
    with open('dataset.jsonl', 'w', encoding="utf-8") as jsonl_obj:
        for  row in csv:
            #if rowCount == 5:
            #   break
            json.dump({'id': rowCount,'created_at': str(row["created_at"]), 'text': row["text"]}, jsonl_obj)
            jsonl_obj.write('\n')
            print(f'RowCount:{rowCount}')
            rowCount = rowCount + 1
