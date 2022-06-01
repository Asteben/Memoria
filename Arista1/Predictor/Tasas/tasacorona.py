from csv import DictReader

with open('CoronaTweetsLimpio2.csv', 'r', encoding="utf-8") as csv_obj:
    csv = DictReader(x.replace('\0', '') for x in csv_obj)
    rowCount = 0
    rowCountCorona = 0
    for  row in csv:
        print(rowCount)
        #if rowCount == 1000000:
        #    break
        resultado = []
        llaves = ('corona', 'covid')
        for coincidencia in llaves:
            if coincidencia in row['text'].lower():
                resultado.append(True)
            else:
                resultado.append(False)
        if resultado[0] or resultado[1]:
            rowCountCorona = rowCountCorona + 1
        rowCount = rowCount + 1
    porcentaje = rowCountCorona / (rowCount/100)
    print(f'Numero de filas leidas: {rowCount}')
    print(f'Numero de filas que presentan la palabra corona o covid: {rowCountCorona}')
    print(f'Porcentaje respectivo de filas leidas: {porcentaje}')
