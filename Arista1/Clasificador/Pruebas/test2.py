import json

with open('../dataset/dataset.jsonl', 'r',  encoding="utf-8") as data1:
    json_list = list(data1)
    rowCount = 1
    rowCountCorona = 0
    with open('../dataset/dataset2.jsonl', 'w', encoding="utf-8") as jsonl_obj:
        for json_str in json_list:
            line = json.loads(json_str)
            #if rowCount == 5:
            #   break
            resultado = []
            llaves = ('corona', 'covid')
            for coincidencia in llaves:
                if coincidencia in line['text'].lower():
                    resultado.append(True)
                else:
                    resultado.append(False)
            if resultado[0] or resultado[1]:
                json.dump({'id': rowCount,'created_at': str(line["created_at"]), 'text': line["text"], 'mark': 1}, jsonl_obj)
                rowCountCorona = rowCountCorona + 1
            else:
                json.dump({'id': rowCount,'created_at': str(line["created_at"]), 'text': line["text"], 'mark': 0}, jsonl_obj)
            jsonl_obj.write('\n')
            print(f'RowCount:{rowCount}')
            rowCount = rowCount + 1
    porcentaje = rowCountCorona / (rowCount/100)
    print(f'Numero de filas leidas: {rowCount}')
    print(f'Numero de filas que presentan la palabra corona o covid: {rowCountCorona}')
    print(f'Porcentaje respectivo de filas leidas: {porcentaje}%')