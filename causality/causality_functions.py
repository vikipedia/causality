import time
import random
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression
import altair as alt
import pytest


def jitter(x, scale=1):
    return x + np.random.normal(scale=scale)


def dependent(x, m, c, error=1):
    return jitter(m*x + c, scale=error)  # mx +c + error


def check_generation(generation, d):
    return d > 0 and generation >= d


def mkfolder(pathway, samplesize, dg, d1, d2, d3, d4):
    tokens = [f"{pathway.__name__}",
              f"samplesize-{samplesize}"]
    if dg < 1:
        tokens.append("dg-{}".format(dg))
    if d1 > 0:
        tokens.append("d1-{}".format(d1))
    if d2 > 0:
        tokens.append("d2-{}".format(d2))
    if d3 > 0:
        tokens.append("d3-{}".format(d3))
    if d4 > 0:
        tokens.append("d4-{}".format(d4))

    path = os.path.join(*tokens)
    os.makedirs(path, exist_ok=True)
    return path


def simulations_data(pathway, n=1000, run=0):
    def save_data(r):
        folder = mkfolder(pathway, samplesize, dg, d1, d2, d3, d4)
        np.savetxt(os.path.join(folder, f"sample{run}.csv"), r, delimiter=",")

    r = np.array([pathway() for i in range(n)])
    # save_data(r)
    return r


def regress(X, Y):
    model = LinearRegression()
    mXY = model.fit(X.reshape(-1, 1), Y)
    r_sqr = mXY.score(X.reshape(-1, 1), Y)
    residual = Y - model.predict(X.reshape(-1, 1))
    return mXY.intercept_, mXY.coef_[0], r_sqr, residual


def get_slope_intercept(model):
    return model._slopt, model._intercept


def compute_regression(ABC):
    A, B, C = ABC.transpose()
    RAB = regress(A, B)
    RBC = regress(B, C)
    RAC = regress(A, C)
    corrE = np.corrcoef(np.array([RAB[3], RBC[3]]))
    corrE_BA_C = np.corrcoef(np.array([RAB[3], C]))

    # print(RAB[1]*RBC[1]-RAC[1]) ## better to look at distribution of this error..it should come with center as 0
    return {"kAB": RAB[0], "kBC": RBC[0], "kAC": RAC[0],
            "mAB": RAB[1], "mBC": RBC[1], "mAC": RAC[1],
            "r_sqrAB": RAB[2], "r_sqrBC": RBC[2], "r_sqrAC": RAC[2],
            "r_E": corrE[0, 1],
            "r_E_BA_C": corrE_BA_C[0, 1],
            "n": len(A)}


def compute_correlation(ABC):
    corr = np.corrcoef(ABC.transpose())
    rAB, rBC, rAC = corr[0, 1], corr[1, 2], corr[0, 2]
    # print(rAB**2*rBC**2-rAC**2) ## better to look at distribution of this error..it should come with center as 0
    # or see correlation between these two quantities should be 1 and if we regress ,
    # it should have slope 1
    return {"rAB": rAB, "rBC": rBC, "rAC": rAC}


def compute_confidence_interval(r, n):
    def boundary(zeta):
        return ((np.exp(2*zeta))-1)/((np.exp(2*zeta))+1)

    z = 0.5*np.log(((1+r)/(1-r)))  # check this
    zetal = z-1.96*np.sqrt(1/(n-3))
    rl = boundary(zetal)
    zetau = z+1.96*np.sqrt(1/(n-3))
    ru = boundary(zetau)
    return rl, ru


def test_compute_confidence_interval():
    pass


def confidence_status(L, U, v, debug=False):
    if debug:
        for l, u, m in zip(L, U, v):
            print(l, u, m)

    l = pd.Series([""]*len(v))
    l = l.mask(v < L, "less")
    if debug:
        print(l)
    l.mask(v > U, "more", inplace=True)
    if debug:
        print(l)
    w = l.mask((v >= L) & (v <= U), "within")
    if debug:
        print((v >= L) & (v <= U))
    return w


def confidence_status_(r, n):
    L, U = compute_confidence_interval(r, n)
    return confidence_status(L, U, r)


def slope_confidence(m, X, Y, k):
    """ as per this pdf
    https://www.ncss.com/wp-content/themes/ncss/pdf/Procedures/PASS/Confidence_Intervals_for_Linear_Regression_Slope.pdf
    """
    n = len(Y)
    Y_ = m*X + k
    numer = np.sqrt(np.sum((Y - Y_)**2)/(n-2))
    denom = np.sqrt(np.sum((X - np.mean(X))**2))
    term = 1.96*numer/denom
    return m-term, m+term


def compute_slope_confidence(m_all, ABC_all, k_all, m_check):
    L, U = [], []

    for m, k, ABC in zip(m_all, k_all, ABC_all):
        A, B, C = ABC.transpose()
        l, u = slope_confidence(m, A, C, k)
        L.append(l)
        U.append(u)

    return confidence_status(pd.Series(L), pd.Series(U), m_check)


def get_value(d, variable):
    if isinstance(variable, str):
        return d[variable]
    else:
        value = d[variable[0]]
        for v in variable[1:]:
            value = value*d[v]
        return value


def compute_confidence_handle_square(d, boundary, variable):
    L, U = compute_confidence_interval(get_value(d, boundary), d['n'])
    conf = confidence_status(L, U, get_value(d, variable))
    sqrdV = get_value(d, variable)**2
    sqrdB = get_value(d, boundary)**2
    outside = conf != "within"
    outside_ = pd.Series([""]*len(outside))
    outside_ = outside_.mask(sqrdV[outside] < sqrdB[outside], "less")
    outside_ = outside_.mask(sqrdV[outside] > sqrdB[outside], "more")

    conf.mask(outside, outside_)
    return conf


def compute_confidence_rAC(d):
    L, U = compute_confidence_interval(
        d.rAC, d['n'])  # whether to remove sqr
    conf_rAC = confidence_status(L, U, d.rAB*d.rBC)
    sqrd = d.rAB**2*d.rBC**2
    sqrAC = d.rAC**2
    outside = conf_rAC != "within"
    outside_ = pd.Series([""]*len(outside))
    outside_ = outside_.mask(sqrd[outside] > sqrAC[outside], "more")
    outside_ = outside_.mask(sqrd[outside] < sqrAC[outside], "less")
    conf_rAC.mask(outside, outside_)
    return conf_rAC


def test_squred(d):
    x = compute_confidence_handle_square(d, 'rAC', ('rAB', 'rBC'))
    assert d['confidence_rAC'].equals(x)


def add_confidence_stats(d, ABC_all):
    d['rAB2*rBC2-rAC2'] = d.rAB**2 * d.rBC**2 - d.rAC**2
    d['r_E_BA_C2-rBC2'] = d.r_E_BA_C**2 - d.rBC**2
    # d['rAC2'] = d.rAC**2
    d['mAB*mBC-mAC'] = d.mAB*d.mBC - d.mAC
    d['mAB*mBC'] = d.mAB*d.mBC
    d['confidence_rAC'] = compute_confidence_rAC(d)
    test_squred(d)
    L, U = compute_confidence_interval(d.r_E, d['n'])
    d['confidence_residual_corr'] = confidence_status(
        L, U, pd.Series(np.zeros_like(L)))
    # L, U = compute_confidence_interval(d.rBC**2, d['n'])  # ???
    d['confidence_corrected_bc_corr'] = compute_confidence_handle_square(d,
                                                                         'rBC',
                                                                         'r_E_BA_C')
    confidence_slope_AC = compute_slope_confidence(d.mAC,
                                                   ABC_all,
                                                   d.kAC,
                                                   d['mAB*mBC'])
    d['confidence_slope_AC'] = confidence_slope_AC


def confidence_charts(folder, d):
    confidence = alt.Chart(d, title=f"Correlation confidence {(d['confidence_rAC']=='within').sum()}/{len(d)}").mark_point().encode(
        x=alt.X('rAB2:Q'),
        y=alt.Y('rBC2:Q'),
        color=alt.Color('confidence_rAC:N',
                        scale=alt.Scale(domain=['less', 'within', 'more'],
                                        range=['orange', 'green', 'red'])),
    ).transform_calculate(
        rAB2='datum.rAB*datum.rAB',
        rBC2='datum.rBC*datum.rBC'
    )

    confidence_res_corr = alt.Chart(d, title=f"residual corr confidence {(d['confidence_residual_corr']=='within').sum()}/{len(d)}").mark_point().encode(
        x=alt.X('rAB2:Q'),
        y=alt.Y('rBC2:Q'),
        color=alt.Color('confidence_residual_corr:N',
                        scale=alt.Scale(domain=['less', 'within', 'more'],
                                        range=['orange', 'green', 'red'])),
    ).transform_calculate(
        rAB2='datum.rAB*datum.rAB',
        rBC2='datum.rBC*datum.rBC'
    )

    confidence_corrected_bc_corr = alt.Chart(d, title=f"corrected bc corr confidence {(d['confidence_corrected_bc_corr']=='within').sum()}/{len(d)}").mark_point().encode(
        x=alt.X('rAB2:Q'),
        y=alt.Y('rBC2:Q'),
        color=alt.Color('confidence_corrected_bc_corr:N',
                        scale=alt.Scale(domain=['less', 'within', 'more'],
                                        range=['orange', 'green', 'red'])),
    ).transform_calculate(
        rAB2='datum.rAB*datum.rAB',
        rBC2='datum.rBC*datum.rBC'
    )

    confidence_slope_AC = alt.Chart(d, title=f"slope AC confidence {(d['confidence_slope_AC']=='within').sum()}/{len(d)}").mark_point().encode(
        x=alt.X('rAB2:Q'),
        y=alt.Y('rBC2:Q'),
        color=alt.Color('confidence_slope_AC:N',
                        scale=alt.Scale(domain=['less', 'within', 'more'],
                                        range=['orange', 'green', 'red'])),
    ).transform_calculate(
        rAB2='datum.rAB*datum.rAB',
        rBC2='datum.rBC*datum.rBC'
    )

    return confidence_slope_AC, confidence, confidence_res_corr, confidence_corrected_bc_corr

def confidence_graphs(folder, d):
    charts = confidence_charts(folder, d)
    resized = [c.properties(height=400, width=400) for c in charts]
    #slop, corr, res_corr, corrected_bc_cor = resized
    #row1 = alt.hconcat(slop, corr)
    #row2 = alt.hconcat(res_corr, corrected_bc_cor)
    #chart = alt.vconcat(row1, row2).interactive()
    chart = alt.hconcat(*resized)
    chart.save(os.path.join(folder, "charts.png"))
    return chart


def line_plot(ABC):
    pass


def stats_graphs(d):
    slope_histogram = alt.Chart(d).mark_bar().encode(
        x=alt.X('mAB*mBC-mAC:Q', bin=True),
        y='count()').properties(title="slope diff histogram")

    bincount = 100
    ticks = 10
    correlation_graph = alt.Chart(d).mark_bar().encode(
        x=alt.X('rAB2*rBC2-rAC2:Q', bin=True,
                axis=alt.Axis(
                    tickCount=ticks,
                    grid=False)),
        y='count()').properties(
            title="Correlation")
    residual_correlation = alt.Chart(d).mark_bar().encode(
        x=alt.X('r_E:Q', bin=True),
        y='count()').properties(title="Correlation of residuals")
    corrected_correlation = alt.Chart(d).mark_bar().encode(
        x=alt.X("r_E_BA_C2-rBC2:Q", bin=True),
        y='count()').properties(
        title="Corrected Correlation")

    row1 = alt.hconcat(slope_histogram, correlation_graph)
    row2 = alt.hconcat(residual_correlation, corrected_correlation)
    chart = alt.vconcat(row1, row2).interactive()
    return chart
