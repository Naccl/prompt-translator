import csv

input_file = 'danbooru.csv'
output_file = 'custom.csv'

with open(input_file, 'r', encoding='utf-8') as file:
    reader = csv.reader(file)
    rows = []
    for row in reader:
        if '|' in row[0]:
            keys = row[0].split('|')
            translation = row[1]
            for key in keys:
                rows.append([key, translation])
        else:
            rows.append(row)

with open(output_file, 'w', encoding='utf-8', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(rows)
