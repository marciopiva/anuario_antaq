import os
import csv
import sqlite3

conn_db = sqlite3.connect('../db/antaq.db')
cursor_db = conn_db.cursor()

path_to_load = os.getcwd() + '../data/to_load'
base_to_load = path_to_load +'../data/base'
years_to_load = path_to_load + '../data/years.txt'

# loading base
for csvfiles_to_load in os.listdir(base_to_load):

    row_count = 0
    ext_pos = csvfiles_to_load.find('.')
    csvfiles_to_load_path = base_to_load + '/' + csvfiles_to_load

    print(f'Loading {csvfiles_to_load_path}...')

    with open(csvfiles_to_load_path) as csvfile_to_load:
        csvreader_file_to_load = csv.reader(csvfile_to_load, delimiter=';')

        csvreader_file_to_load.__next__()
        for csvfile_to_load_row in csvreader_file_to_load:
            col_count = 0
            str_insert = 'insert into ' + csvfiles_to_load[:ext_pos] + ' values ( '

            for col in csvfile_to_load_row:
                str_insert = str_insert + '\'' + col.replace('\'','') + '\''
                if col_count < len(csvfile_to_load_row)-1:
                    str_insert = str_insert + ', '
                col_count += 1
            str_insert = str_insert + ');'

            cursor_db.execute(str_insert)

            row_count += 1
            if row_count % 10000 == 0:
                print(f'\t{row_count} rows inserted ... commit!')
                conn_db.commit()


        conn_db.commit()
        print(f'Total rows of {csvfiles_to_load_path}: {row_count}')


# loading years
with open(years_to_load) as csvyears:
    csvreader_years = csv.reader(csvyears, delimiter=';')

    csvreader_years.__next__()
    for csvyears_row in csvreader_years:

        for csvfiles_to_load in os.listdir(path_to_load + '/' + csvyears_row[0]):

            csvfiles_to_load_path = path_to_load + '/' + csvyears_row[0] + '/' + csvfiles_to_load
            with open(csvfiles_to_load_path) as csvfile_to_load:

                row_count = 0
                ext_pos = csvfiles_to_load.find('.')
                csvreader_file_to_load = csv.reader(csvfile_to_load, delimiter=';')
                
                csvreader_file_to_load.__next__()
                for csvfile_to_load_row in csvreader_file_to_load:

                    col_count = 0
                    str_insert = 'insert into ' + csvfiles_to_load[:ext_pos] + ' values ( '

                    for col in csvfile_to_load_row:
                        str_insert = str_insert + '\'' + col.replace('\'','') + '\''
                        if col_count < len(csvfile_to_load_row)-1:
                            str_insert = str_insert + ', '
                        col_count += 1
                    str_insert = str_insert + ');'

                    cursor_db.execute(str_insert)

                    row_count += 1
                    if row_count % 10000 == 0:
                        print(f'\t{row_count} rows inserted ... commit!')
                        conn_db.commit()

                conn_db.commit()
                print(f'Total rows of {csvfiles_to_load_path}: {row_count}')

conn_db.close()


