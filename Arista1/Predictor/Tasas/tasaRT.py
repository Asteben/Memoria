from csv import DictReader

with open('12MCoronaTweets.csv', 'r', encoding="utf-8") as csv_obj:
    csv = DictReader(x.replace('\0', '') for x in csv_obj)
    rowCount = 0
    rowCountRT = 0
    for  row in csv:
        print(rowCount)
        #if rowCount == 500000:
        #  break
        resultado = []
        llaves = ('RT')
        for coincidencia in llaves:
            if coincidencia in row['text']:
                resultado.append(True)
            else:
                resultado.append(False)
        if resultado[0]:
            rowCountRT = rowCountRT + 1
        rowCount = rowCount + 1
    porcentaje = rowCountRT / (rowCount/100)
    print(f'Numero de filas leidas: {rowCount}')
    print(f'Numero de filas que presentan un RT: {rowCountRT}')
    print(f'Porcentaje respectivo de filas leidas: {porcentaje}')