# Data folder

Place the two partitioned UNSW-NB15 files here (names must match):

- `UNSW_NB15_training-set.csv`  (~175,341 rows)
- `UNSW_NB15_testing-set.csv`   (~82,332 rows)

Sources:
- Official: https://research.unsw.edu.au/projects/unsw-nb15-dataset
- Kaggle mirror: https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15

Do NOT use the four raw `UNSW-NB15_1..4.csv` files for this project; they have no header
row and a different layout. If you only have the real data elsewhere, pass paths explicitly:

    python -m src.train --train /path/train.csv --test /path/test.csv
