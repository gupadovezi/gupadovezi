# -*- coding: utf-8 -*-
"""Test_t_anova_python.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1SvWZao1FFnpWIWCrcFcXYYRPfN79uY2p
"""

pip install scipy

from scipy.stats import ttest_ind

# Exemplo de dados: grupo A e grupo B
grupo_a = [5.1, 5.5, 5.7, 6.0, 5.8]
grupo_b = [6.2, 6.5, 6.3, 6.7, 6.1]

# Teste t de duas amostras independentes
t_stat, p_value = ttest_ind(grupo_a, grupo_b)

print("Estatística t:", t_stat)
print("Valor-p:", p_value)

from scipy.stats import ttest_rel

# Medidas antes e depois de uma intervenção
antes = [120, 115, 130, 110, 125]
depois = [112, 110, 125, 105, 120]

# Teste t pareado
t_stat, p_value = ttest_rel(antes, depois)

print("Estatística t:", t_stat)
print("Valor-p:", p_value)

from scipy.stats import ttest_1samp

# Dados da amostra
dados = [22, 21, 23, 20, 25, 24]

# Testar se a média é diferente de 20
t_stat, p_value = ttest_1samp(dados, popmean=20)

print("Estatística t:", t_stat)
print("Valor-p:", p_value)

pip install scipy statsmodels pandas

from scipy.stats import f_oneway

# Três grupos com diferentes amostras
grupo1 = [5.1, 5.3, 5.5, 5.7]
grupo2 = [6.2, 6.5, 6.1, 6.4]
grupo3 = [7.0, 6.8, 7.2, 7.1]

# ANOVA de uma via
f_stat, p_value = f_oneway(grupo1, grupo2, grupo3)

print("Estatística F:", f_stat)
print("Valor-p:", p_value)

import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

# Exemplo de dados
data = pd.DataFrame({
    'valor': [5.1, 5.3, 5.5, 5.7, 6.2, 6.5, 6.1, 6.4, 7.0, 6.8, 7.2, 7.1],
    'grupo': ['A']*4 + ['B']*4 + ['C']*4
})

# Modelo OLS
modelo = ols('valor ~ grupo', data=data).fit()

# ANOVA
anova_resultado = anova_lm(modelo)

print(anova_resultado)

pip install pingouin

import pingouin as pg
import pandas as pd

# Exemplo: 3 condições medidas no mesmo sujeito
data = pd.DataFrame({
    'subject': [1, 2, 3, 4, 5],
    'cond1': [4, 5, 6, 5, 4],
    'cond2': [5, 6, 7, 6, 5],
    'cond3': [6, 7, 8, 7, 6]
})

# Reshape the data from wide to long format
# This is necessary for pingouin.rm_anova when you have multiple levels of a single within-subject factor
data_long = pd.melt(data, id_vars=['subject'], var_name='condition', value_name='value')

# Now perform the repeated measures ANOVA on the long-format data
# dv is the dependent variable ('value')
# within is the within-subject factor ('condition')
# subject is the subject identifier ('subject')
anova = pg.rm_anova(data=data_long, dv='value', within='condition', subject='subject')

print(anova)