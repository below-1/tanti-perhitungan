import random
import numpy as np
import pandas as pd
import itertools

data = [
  (
    'H1',
    {3, 1}
  ),
  (
    'H2',
    {6, 8, 9}
  ),
  (
    'H8',
    {22, 23, 24}
  ),
  (
    'P5',
    {44, 45, 47}
  ),
  (
    'H1',
    {3, 4}
  ),
  (
    'P6',
    {48, 50}
  ),
  (
    'P9',
    {58}
  ),
  (
    'P7',
    {52, 54}
  ),
  (
    'H9',
    {25, 26}
  ),
  (
    'H4',
    {12, 13}
  ),
  (
    'P2',
    {35, 36}
  ),
  (
    'P8',
    {56, 57}
  ),
  ('P12', {65, 66}),
  ('P13', {67, 68, 69, 70}),
  ('H1', {3, 2, 1}),
  ('H3', {10}),
  ('H4', {13, 14}),
  ('P3', {38, 39}),
  ('H1', {1, 5}),
  ('H5', {15, 16}),
  ('H5', {15, 17}),
  ('P1', {31, 32, 33}),
  ('H10', {27, 29}),
  ('P11', {64}),
  ('P10', {60, 62}),
  ('P6', {48, 49, 50}),
  ('P9', {59}),
  ('H9', {25}),
  ('P13', {67, 68}),
  ('P7', {53, 54})
]

# Encode all Target
targets = list({ t for t, _ in data })
def _sort_target(t):
    c, *index = t
    return (c, int(''.join(index)))
targets.sort(key=_sort_target)

# encode all attributes
attributes = list(set( attr for (_, attrs) in data for attr in attrs   ))
attributes.sort()

rules = [
    { 'H1': [1, 2, 3, 4, 5] },
    { 'H2': [6, 7, 8] },
    { 'P13': [ 67, 68, 69, 70 ] },
    { 'P6': [45, 46, 47] }
]
data2 = [
    (1, { 1, 2, 3 }),
    (1, { 1, 2, 4 }),
    (1, { 3, 5 }),
    (2, { 6, 7, 8 }),
    (2, { 7, 8 }),
    (3, { 67, 68, 69 }),
    (3, { 67, 70 }),
    (3, { 69, 70 }),
    (4, { 45, 46 }),
    (4, { 45, 46, 47 })
]
targets = list({ t for t, _ in data2 })
targets.sort()
attributes = list(set( attr for (_, attrs) in data2 for attr in attrs   ))
attributes.sort()

N_P = len(targets)
N_G = len(attributes)
N_CASE = 30
PENYAKIT_LIST = list(range(1, N_P + 1))
GEJALA_LIST = list(range(1, N_G + 1))
GEJALA_HEADERS = [ f'G{i}' for i in GEJALA_LIST  ]
COLUMN_HEADERS = ['P'] + GEJALA_HEADERS

class Case:
    def __init__(self, p, gs):
        self.p = p
        self.gs = gs

    def __repr__(self):
        return 'Case(p={}, gs={})\n'.format(self.p, self.gs)

def to_list(case):
    p = case.p
    _gs = case.gs
    gs = [ 1 if g in _gs else 0 for g in GEJALA_LIST ]
    return [ p, *gs ]

def build_case_df2():
    targets = list(rules.keys())

def build_case_df():
    _list = []
    for _target, _attrs in data2:
        target = targets.index(_target)
        attrs = [ 1 if attributes[i - 1] in _attrs else 0 for i in GEJALA_LIST ]
        row = [ target, *attrs ]
        _list.append(row)
    case_df = pd.DataFrame(_list, columns=COLUMN_HEADERS)

    return case_df

def build_count_summaries(df):
    total_data = df.shape[0] * 1.0
    group_by_p = df.groupby('P')

    def count_Zero_One(column, gdf): 
        return [ 
            gdf[gdf[column] == 0].shape[0],
            gdf[gdf[column] == 1].shape[0]
        ]
    
    def prob_penyakit(g):
        return g.shape[0] / df.shape[0] * 1.0

    result = []
    for name, group in group_by_p:
        length = group.shape[0]
        row = [ group.iloc[0][0], prob_penyakit(group) ]
        for gh in GEJALA_HEADERS:
            for occ in count_Zero_One(gh, group):
                row.append(occ)
        result.append(row)
    return np.array(result)

def build_nb_summaries(df):
    total_data = df.shape[0] * 1.0
    group_by_p = df.groupby('P')

    def count_Zero_One(column, gdf): 
        return [ 
            gdf[gdf[column] == 0].shape[0],
            gdf[gdf[column] == 1].shape[0]
        ]
    
    def prob_penyakit(g):
        return g.shape[0] / df.shape[0] * 1.0

    result = []
    for name, group in group_by_p:
        length = group.shape[0]
        row = [ group.iloc[0][0], prob_penyakit(group) ]
        for gh in GEJALA_HEADERS:
            for occ in count_Zero_One(gh, group):
                g_prob = (occ + 1) / (length + 1)
                row.append(g_prob)
        result.append(row)
    return np.array(result)

def test_case(case, count_summary, summary, original_data):
    _gs = case.gs
    p = case.p
    gs = [ 1 if g in _gs else 0 for g in GEJALA_LIST ]
    gs = [ (g, v) for g, v in zip(GEJALA_HEADERS, gs) ]
    _filter = [ ('PENY', ''), ('P(PENY)', ''), *gs ]
    filtered = summary[ _filter ]
    prob = filtered.iloc[:, 2:].apply(np.prod, axis=1) * filtered['P(PENY)']
    filtered['case_prob'] = prob

    writer = pd.ExcelWriter('tanti-perhitungan.xlsx')
    original_data.to_excel(writer, 'DATA')
    count_summary.to_excel(writer, 'COUNT SUMMARY')
    summary.to_excel(writer, 'PROB SUMMARY')
    filtered.to_excel(writer, 'HASIL NAIVE BAYES')
    writer.save()


if __name__ == '__main__':
    case_df = build_case_df()
    count_summaries = build_count_summaries(case_df)
    summaries = build_nb_summaries(case_df)

    tuples = list(itertools.product(GEJALA_HEADERS, [0, 1]))
    tuples = [ ('PENY', ''), ('P(PENY)', '') ] + tuples
    index = pd.MultiIndex.from_tuples(tuples)

    row_headers = [f'P{p}' for p in PENYAKIT_LIST]
    summary_df = pd.DataFrame(summaries, index=row_headers, columns=index)
    count_summary_df = pd.DataFrame(count_summaries, index=row_headers, columns=index)
    # Change first column to int32
    summary_df['PENY'] = summary_df['PENY'].astype(np.int32)

    rand_case = Case(p=1, gs=[6, 67, 8])

    test_case(rand_case, count_summary_df, summary_df, case_df)