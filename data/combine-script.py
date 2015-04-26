import csv

classifiers = ['jogging','walking','upstairs','downstairs']

def combineCSVs(classifiers):
    '''combine CSVs takes a list of classifiers, grabs the csvs, and combines them, adding a column for activity names'''

    outfile = open('combined-data.csv','a')

    for classifier in classifiers:
        infile = open(classifier+'.csv', 'rb')
        temp = open(classifier+'update.csv','wb')
        csvreader = csv.DictReader(infile)
        fieldnames=['Activity'] + csvreader.fieldnames #add column for Activities at beginnning
        csvwriter = csv.DictWriter(temp, fieldnames)
        csvwriter.writeheader()

        for activity, row in enumerate(csvreader, 1):
            csvwriter.writerow(dict(row,Activity=classifier))

        outfile.write(temp)
        outfile.close()
    return outfile
print combineCSVs(classifiers)