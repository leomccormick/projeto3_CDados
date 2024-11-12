import statsmodels.api as sm

def regress(Y, X):
    '''
    Y: coluna do DataFrame utilizada como variável resposta (TARGET)
    X: coluna(s) do DataFrame utilizadas como variável(is) explicativas (FEATURES)
    '''
    X_cp = sm.add_constant(X)
    model = sm.OLS(Y,X_cp)
    results = model.fit()

    return results


def pega_variaveis(Y, X, corte):
    results = regress(Y, X)
    while results.pvalues.max() > corte:
        maior = results.pvalues.tolist().index(results.pvalues.max())
        X = X.drop(columns=[list(X)[maior-1]])
        results = regress(Y, X)
    return X


