import sqlite3

conn_db = sqlite3.connect('../db/antaq.db')
cursor_db_select = conn_db.cursor()
cursor_db_delete = conn_db.cursor()

select_tables = 'SELECT name FROM sqlite_master WHERE type= ' + '\'table\''
for row in cursor_db_select.execute(select_tables):
    print(f'truncating table: {row[0]}...')
    cursor_db_delete.execute('DELETE FROM ' + row[0] + ';')
    conn_db.commit()

conn_db.close()
