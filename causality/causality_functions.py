import time
import random
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import altair as alt


def jitter(x, scale=1):
    return x + np.random.normal(scale=scale)


def dependent(x, m, c, error=1):
    return jitter(m*x + c, scale=error)  # mx +c + error


def simulations_data(pathway, n=1000):
    random.seed(time.time())
    return np.array([pathway() for i in range(n)])


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
    corrE_BA_C = np.corrcoef(np.array([np.square(RAB[3]), C]))

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

    z = 0.5*np.log(((1+r)/(1-r)))# check this
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


def slope_confidence(m, X, Y):
    """ as per this pdf
    https://www.ncss.com/wp-content/themes/ncss/pdf/Procedures/PASS/Confidence_Intervals_for_Linear_Regression_Slope.pdf
    """
    n = len(Y)
    numer = np.sqrt(np.sum((Y - np.mean(Y))**2)/(n-2))
    denom = np.sqrt(np.sum((X - np.mean(X))**2))
    term = 1.96*numer/(denom*(n-2)**0.5)
    return m-term, m+term


def compute_slope_confidence(m_all, ABC_all):
    L, U = [], []

    for m, ABC in zip(m_all, ABC_all):
        A, B, C = ABC.transpose()
        l, u = slope_confidence(m, A, C)
        L.append(l)
        U.append(u)

    return confidence_status(pd.Series(L), pd.Series(U), m_all)


def add_confidence_stats(d, ABC_all):
    d['rAB2*rBC2-rAC2'] = d.rAB**2 * d.rBC**2 - d.rAC**2
    d['r_E_BA_C2-rBC2'] = d.r_E_BA_C**2 - d.rBC**2
    # d['rAC2'] = d.rAC**2
    d['mAB*mBC-mAC'] = d.mAB*d.mBC - d.mAC
    L, U = compute_confidence_interval(d.rAC**2, d['n']) # whether to remove sqr
    d['confidence_rAC'] = confidence_status(L, U, d.rAB**2*d.rBC**2)
    L, U = compute_confidence_interval(d.r_E, d['n'])
    d['confidence_residual_corr'] = confidence_status(
        L, U, pd.Series(np.zeros_like(L)))
    L, U = compute_confidence_interval(d.rBC**2, d['n'])  # ???
    d['confidence_corrected_bc_corr'] = confidence_status(L, U, d.r_E_BA_C**2)
    d['confidence_slope_AC'] = compute_slope_confidence(d.mAC, ABC_all)


def confidence_graphs(d):
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
    return alt.vconcat(confidence_slope_AC, confidence, confidence_res_corr, confidence_corrected_bc_corr)


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

    return alt.vconcat(slope_histogram, correlation_graph, residual_correlation, corrected_correlation)
