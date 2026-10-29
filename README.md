# Day 3 - Statistical Analysis Toolkit 

## How to run

```bash
pip install -r requirements.txt
python stats_toolkit.py          
python -m unittest -v            
```

`python stats_toolkit.py` also runs the unit tests and pastes their result into
the report. You can pass another CSV path as the first argument.

## Approach

 `load_column` reads the CSV once and returns the valid floats plus a list of
  `(row_number, raw_value)` rejections. Row numbers are file line numbers
  the header is line 1, so row 7 is line 7 in a text editor.
 `StatisticsAnalyzer` sorts the data once in `__init__` and caches every
  result, so repeated calls never re-sort or recompute. `without_outliers()`
  returns a new analyzer and leaves the original untouched.
 `from_csv` also keeps each value's `date` and `agent_id`, so the report can
  say which day and agent an outlier came from.

## Choices

Sample vs population variance: sample (n - 1). The 40 days are a slice of
an ongoing process Sept-Oct 2026, not every day the agents will ever trade, so
we use them to estimate the underlying spread. Sample variance corrects the
small-sample bias of dividing by n. Both versions are available through
`sample=True/False`.

Quartile method: linear interpolation (Hyndman & Fan type 7).Position is
`(n - 1) * p` and the value is interpolated between the two nearest ranks. This
is NumPy's `method="linear"` its default and Excel's `QUARTILE.INC`. The
NumPy comparison passes `method="linear"` explicitly so both sides use the same
method.

What counts as invalid: rejected, never guessed or repaired

 empty / whitespace, missing value 
 `-`, `n/a`, `na`, `null`, `none`, `nan`, `?`, `x`, `--`  
 placeholder, not a number 
 anything not matching plain digits with one optional decimal point, e.g. `12,40O`  
 not a valid number 
 negative numbers, impossible deposit 
`12,40O` looks like a typo for 12,400 but we cannot know that, so it is
rejected. Thousands separators like `12,400` are also rejected for the same
reason. Extreme values such as `96500.00` and `1200.00` are
kept: they may be real and the IQR step is what flags them.


## Assumptions

The column of interest is `daily_deposits_kes`; the `transactions` column
(which also has junk such as `abc`) is not analysed.
Deposits cannot be negative, zero is allowed.
Dates and agent IDs are used only as labels not validated.
Outlier values are flagged for review not deleted from the source file.
