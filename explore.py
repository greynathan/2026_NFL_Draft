import nflreadpy as nfl
import pandas as pd

df = nfl.import_pbp_data([2025], ['reg'], downcast=True)

df.head()