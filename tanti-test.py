import random
import numpy as np
import pandas as pd
import itertools

N_P = 3
N_G = 6
N_CASE = 10
PENYAKIT_LIST = list(range(1, N_P + 1))
GEJALA_LIST = list(range(1, N_G + 1))
GEJALA_HEADERS = [ f'G{i}' for i in GEJALA_LIST  ]
COLUMN_HEADERS = ['P'] + GEJALA_HEADERS
print(GEJALA_HEADERS)

class Case:
    def __init__(self, p, gs):
        self.p = p
        self.gs = gs

    def __repr__(self):
        return 'Case(p={}, gs={})\n'.format(self.p, self.gs)

def random_case():
    p = random.choice(PENYAKIT_LIST)
    n_g = random.randrange(1, 5)
    gs = frozenset( random.sample(GEJALA_LIST, n_g) )
    return Case(p, gs)

def to_list(case):
    p = case.p
    _gs = case.gs
    gs = [ 1 if g in _gs else 0 for g in GEJALA_LIST ]
    return [ p, *gs ]

def build_case_df():
    case_base = [ random_case() for i in range(N_CASE) ]
    case_array = np.array([ to_list(case) for case in case_base ])
    case_df = pd.DataFrame(case_array, columns=COLUMN_HEADERS)

    return case_df

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

def test_case(case, summary, original_data):
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
    summary.to_excel(writer, 'SUMMARY')
    filtered.to_excel(writer, 'HASIL NAIVE BAYES')
    writer.save()


if __name__ == '__main__':
    case_df = build_case_df()
    summaries = build_nb_summaries(case_df)

    tuples = list(itertools.product(GEJALA_HEADERS, [0, 1]))
    tuples = [ ('PENY', ''), ('P(PENY)', '') ] + tuples
    index = pd.MultiIndex.from_tuples(tuples)

    row_headers = [f'P{p}' for p in PENYAKIT_LIST]
    summary_df = pd.DataFrame(summaries, index=row_headers, columns=index)
    # Change first column to int32
    summary_df['PENY'] = summary_df['PENY'].astype(np.int32)

    rand_case = Case(p=1, gs=[1, 4, 5])

    test_case(rand_case, summary_df, case_df)