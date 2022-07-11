from csv import DictReader

def count(raw,clean):

    with open(raw, 'r', encoding="utf-8") as csv_obj:
        csv = DictReader(x.replace('\0', '') for x in csv_obj)
        rowCount = 0
        '''
        for  row in csv:
            rowCount = rowCount + 1
            '''
        with open(clean, 'r', encoding="utf-8") as csv_obj:
            csv_clean = DictReader(x.replace('\0', '') for x in csv_obj)
            rowCount_clean = 0
            row_covid = 0
            row_other = 0
            for  row in csv_clean:
                rowCount_clean = rowCount_clean + 1
                if row['label'] == 'covid':
                    row_covid = row_covid+1
                elif row['label'] == 'other':
                    row_other = row_other+1
            print(f'Numero de filas covid: {row_covid}')
            print(f'Numero de filas other: {row_other}')
        '''
        porcentaje = rowCount_clean / (rowCount/100)
        print(f'Numero de filas raw: {rowCount}')
        print(f'Numero de filas clean: {rowCount_clean}')
        print(f'Porcentaje respecto de raw: {porcentaje}')
        '''

count('../Data/full_data/12MCoronaTweets.csv','../Data/raw_data/CovidTweetsClean.csv')
'''
count('../Data/full_data/harvey-irma-maria_hydrated.csv','../Data/raw_data/HurricaneTweetsClean.csv')
count('../Data/full_data/nepal_hydrated.csv','../Data/raw_data/EarthquakeTweetsClean.csv')'''