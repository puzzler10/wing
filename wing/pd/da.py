def da(df, nrow = 100):
    """Display all columns in a pandas DataFrame.
    nrow: the number of rows to show"""
    with pd.option_context("display.max_rows", nrow, "display.max_columns", 1000): 
        display(df)