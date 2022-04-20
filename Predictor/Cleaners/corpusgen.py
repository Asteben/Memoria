from csv import DictReader, DictWriter
import time
import datetime

with open('../../Data/rawdataset/CoronaTweetsLimpio3.csv', 'r', encoding="utf-8") as csv_obj:
    csv = DictReader(x.replace('\0', '') for x in csv_obj)
    csv = sorted(csv, key = lambda row: row['created_at'])      #Se ordena de manera ascendente en función de la fecha de los tweets

    rowCount = 0                #Contador de filas a agrupar
    rowCountTotal = 0           #Contador de filas totales del dataset original
    Seconds = 10                #Cantidad de segundos del bloque de tiempo
    UnixGroup = 0               #Bloque de tiempo en formato Unix (son segundos)

    with open((f'../../Data/dataset/dataset_groupby_{Seconds}s.csv'), 'w', encoding="utf-8", newline='') as csv_obj_w:
        csvf = DictWriter(csv_obj_w, fieldnames = ['Unix', 'Quantity'])
        #csvf.writerow(dict((fn,fn) for fn in csv.fieldnames))
        #csvf.writeheader()

        for  row in csv:
            rowCount = rowCount + 1
            rowCountTotal = rowCountTotal + 1
            Unixinrow = int(row['created_at'])          #Fecha en segundos del tweet actual

            if rowCount == 1:
                UnixGroup = (int(Unixinrow/Seconds))*Seconds    #Se define el primer bloque de tiempo en funcion del primer tweet
                #UnixGroup = Unixinrow
            
            if Unixinrow - UnixGroup > Seconds :                #Si la fecha del tweet actual se pasa del bloque de tiempo, se agregan los tweets anteriores al bloque actual
                csvf.writerow({'Unix' : int(UnixGroup), 'Quantity' : (rowCount-1)})
                UnixGroup = UnixGroup + Seconds                 #Se actualiza el bloque de tiempo
                while Unixinrow - (UnixGroup) > Seconds :       #Se agregan 0s mientras la fecha del tweet actual siga siendo mayor al siguiente bloque de tiempo
                    csvf.writerow({'Unix' : int(UnixGroup), 'Quantity' : 0})
                    UnixGroup = UnixGroup + Seconds
                rowCount = 1

            print(f'RowCount:{rowCount}     RowCountTotal:{rowCountTotal}')

            #if rowCountTotal == 500:
            #   break
        csvf.writerow({'Unix' : int(UnixGroup), 'Quantity' : (rowCount)}) #Ultimo grupo