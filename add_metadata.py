"""
DataFrame -> Series
Takes the "date" column of the dataframe, assuming it is in the format of an integer
YYYYMMDD and divides by 10000 to get just the year, returning a Series of the year
"""
def make_year_column(speeches):
    year_col = speeches["date"] // 10000
    return year_col

"""
DataFrame -> Series
Takes the "year" column and makes a grouping for each n-year interval. If offset is 0, 
the grouping restarts whenever speeches["year"] % n == 0, otherwise, it starts whenever
the modulo is equal to the offset. 
"""
def make_n_year_groupings(speeches, n=5, offset=0):
    grouping_col = (speeches["year"] - offset) // n
    return grouping_col

"""
DataFrame Int -> Series
Non mutating: creates a new column that groups every n columns into its own category
"""
def gather_into_groups_of_n(frame, n=100):
    groupings = pd.Series(range(len(frame)), index=frame.index)
    groupings = groupings // n
    
    return groupings