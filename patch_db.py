import sqlite3
conn = sqlite3.connect('eclipse.db')
cols = [
    ('depth_ppm', 'FLOAT'),
    ('rp_rearth', 'FLOAT'),
    ('t_eq_kelvin', 'FLOAT'),
    ('n_transits', 'INTEGER'),
    ('period_lower', 'FLOAT'),
    ('period_upper', 'FLOAT')
]
for col, type_ in cols:
    try:
        conn.execute(f'ALTER TABLE candidates ADD COLUMN {col} {type_};')
        print(f'Added {col}')
    except Exception as e:
        print(f'Skipped {col}: {e}')
conn.commit()
conn.close()
