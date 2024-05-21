import pandas as pd
from pathlib import Path
import numpy as np
from collections import defaultdict

from typing import Optional, List


MOVEMENTS = ('standing', 'walking', 'trotting', 'galloping')
DATA_ROOT = Path('datasets/HorsingAround/csv')


def get_activity_distribution(data_root: Path, movements: List[str]) -> pd.DataFrame:
    activity_distribution = pd.read_csv(data_root / 'activity_distribution.csv')
    columns_of_interest = []
    for movement in movements:
        for column in activity_distribution.columns:
            if column.startswith(movement):
                columns_of_interest.append(column)
    activity_distribution = activity_distribution[['Row'] + columns_of_interest]
    return activity_distribution


def get_horses_of_interest(data_root: Path, movements=MOVEMENTS) -> List[str]:
    activity_distribution = get_activity_distribution(data_root, movements)
    horses_of_interest = []
    for i, row in activity_distribution.iterrows():
        if row['Row'] != 'total':
            movement_counts = defaultdict(float)
            for movement in activity_distribution.columns[1:]:
                movement_type = movement.split('_')[0]
                count = row[movement]
                if count == count:
                    movement_counts[movement_type] += row[movement]
            valid = True
            for movement in movements:
                if movement_counts[movement] < 1.:
                    valid = False
                    break
            if valid:
                horses_of_interest.append(row['Row'])
    return horses_of_interest


def get_horse_dataframes(data_root: Path, horse_name: str) -> List[pd.DataFrame]:
    dataframes = []
    for f in data_root.glob(f"*{horse_name}*"):
        df = pd.read_csv(f, low_memory=False)
        df = df[['label', 'segment', 'Ax', 'Ay', 'Az']].dropna()
        df['norm'] = np.sqrt(df['Ax']**2 + df['Ay']**2 + df['Az']**2)
        if len(df) > 0:
            dataframes.append(df)
            print(len(df))
    return dataframes

if __name__ == '__main__':
    horses = get_horses_of_interest(DATA_ROOT, MOVEMENTS)
    horses_dataframes = defaultdict(list)
    for horse in horses:
        print(horse)
        horses_dataframes[horse].append(get_horse_dataframes(DATA_ROOT, horse))
    print('DONE')
