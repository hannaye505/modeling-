#!/usr/local/bin/python3.6
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import re, csv, sys, os, time ,importlib, logging, pickle, xlrd
from pandas import Series, DataFrame
from flask import Flask, request, render_template
from jinja2 import Environment, FileSystemLoader
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, Imputer
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from scipy import interp
from itertools import chain , cycle
from collections import Counter
from xlutils.copy import copy
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.iolib.smpickle import load_pickle
import statsmodels as statsmodels
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl


pd.set_option('display.max_colwidth', -2)
importlib.reload(sys)


############### Logging #################
def mylog(filename):
    # record running log
    log = logging.getLogger(filename)
    log.setLevel(logging.DEBUG)
    formats = logging.Formatter('%(asctime)s %(filename)s %(funcName)s [line:%(lineno)d] %(levelname)s %(message)s', datefmt='[%Y-%m-%d %H:%M:%S]')
    sh = logging.StreamHandler()
    sh.setFormatter(formats)
    th = logging.FileHandler(filename = os.path.abspath(filename + '.log'), encoding = 'utf-8')
    th.setFormatter(formats)
    log.addHandler(sh)
    log.addHandler(th)
    return log


def colname_suffix(suffix, name):
    # rename columns
    colname = []
    j = 1
    while j <= suffix:
        colname.append('{}{}'.format(name, j))
        j += 1
    return colname


######## Proc Freq ##############
def Proc_Freq(str_var, n, df, logger):
    Freq_str = pd.DataFrame(
        columns=['Variable', 'Value', '频数统计 Count', '总频数百分比 Percent', '累积频数统计 Cum Freq', '累积百分比 Cum Pct'])
    for i in str_var:
        logger.info('The frequency of the "%s" string variable is being counted' % (i))
        tmp = pd.DataFrame(
            columns=['Variable', 'Value', '频数统计 Count', '总频数百分比 Percent', '累积频数统计 Cum Freq', '累积百分比 Cum Pct'])
        length = len(df[i].value_counts())
        tmp['Variable'] = [i] * length
        tmp['Value'] = df[i].value_counts().index
        # count the frequency of string and category variables
        tmp['频数统计 Count'] = df[i].value_counts().get_values()
        tmp['累积频数统计 Cum Freq'] = tmp['频数统计 Count'].cumsum()
        tmp['总频数百分比 Percent'] = (tmp['频数统计 Count'] / sum(tmp['频数统计 Count'])).round(2)
        tmp['累积百分比 Cum Pct'] = (tmp['总频数百分比 Percent'].cumsum()).round(2)
        tmp = tmp.head(n)
        Freq_str = Freq_str.append(tmp, ignore_index=True)
    Freq_str['频数统计 Count'] = Freq_str['频数统计 Count'].astype('int64')
    Freq_str['累积频数统计 Cum Freq'] = Freq_str['累积频数统计 Cum Freq'].astype('int64')
    Freq_str['总频数百分比 Percent'] = Freq_str['总频数百分比 Percent'].astype('float64')
    Freq_str['累积百分比 Cum Pct'] = Freq_str['累积百分比 Cum Pct'].astype('float64')
    # generate the pivot table
    table = pd.pivot_table(Freq_str, index=['Variable', 'Value'],
                           values=['频数统计 Count', '总频数百分比 Percent', '累积频数统计 Cum Freq', '累积百分比 Cum Pct'])
    Proc_Freq_table = table.groupby(level='Variable', group_keys=False).apply(pd.DataFrame.sort_values, by='频数统计 Count',
                                                                         ascending=False)
    return Proc_Freq_table


######## Proc Univariant ##############
def Proc_Uni(num_var, df, logger):
    logger.info('The quantile of the numerical variables is being calculated')
    # calculate statistics for numerical variables
    Uni_num = df[num_var].describe().transpose()
    Uni_num['sum'] = df[num_var].sum().tolist()
    Uni_num['miss'] = len(df[num_var]) - df[num_var].count()
    Uni_num['10%'] = df[num_var].quantile(0.1)
    Uni_num['90%'] = df[num_var].quantile(0.9)
    Uni_num.rename(columns={'count': 'nonmiss'}, inplace=True)
    Uni_num['nonmiss'] = Uni_num['nonmiss'].astype('int64')
    cols = ['nonmiss', 'mean', 'sum', 'miss', 'max', 'min', '10%', '25%', '50%', '75%', '90%']
    Uni_num = Uni_num.loc[:, cols]
    Uni_num = Uni_num.round(2)
    return Uni_num


######## QC Report ##############
def QCreport(libname, filename, n):
    logger = mylog('QCReport')
    file = "{}/{}.csv".format(libname, filename)
    df = pd.read_csv(file, encoding='utf-8')
    df = pd.DataFrame(df)
    allvar = df.columns.tolist()
    # *** get the variable type list  ***
    types = df.dtypes.tolist()
    Proc_contents = {'Variable': allvar, 'Type': types}
    Proc_contents = pd.DataFrame(Proc_contents).to_html(classes='proc')
    df_shape = df.shape
    # ***** Missing Report ***********
    logger.info('The missing values of the QC Report are being counted')
    MissingRec = pd.DataFrame(
        columns=['Variable', '记录缺失数Record Missing', '记录非缺失数Record NonMissing', '缺失百分比 Missing PCT'])
    MissingRec['Variable'] = allvar
    tmp1 = pd.isnull(df).apply(lambda x: x.value_counts()).transpose().fillna(0)
    MissingRec['记录缺失数Record Missing'] = tmp1[True].get_values().astype('int64')
    MissingRec['记录非缺失数Record NonMissing'] = tmp1[False].get_values().astype('int64')
    MissingRec['缺失百分比 Missing PCT'] = ((MissingRec['记录缺失数Record Missing'] / len(df)) * 100).round(2)
    MissingRec['缺失百分比 Missing PCT'] = MissingRec['缺失百分比 Missing PCT'].astype(object)
    MissingRec['缺失百分比 Missing PCT'] = MissingRec['缺失百分比 Missing PCT'].apply(lambda x: '{}%'.format(x))
    tmp = df.dtypes == object
    logger.info('The data is being segmented by data type')
    # distinguish between numerical and string variables
    str_var = df.select_dtypes(include=['object']).columns.tolist()
    rest_var = list(set(allvar).difference(set(str_var)))
    cat_var = []
    for i in rest_var:
        if len(df[rest_var][i].unique().tolist()) <= 3:
            cat_var.append(i)
    num_var = list(set(rest_var).difference(set(cat_var)))
    num_var_full = df[num_var]
    num_name = "{}/num_var_full.csv".format(libname)
    num_var_full.to_csv(num_name, index=False)
    if str_var != []:
        str_var_full = df[str_var]
        str_name = "{}/str_var_full.csv".format(libname)
        str_var_full.to_csv(str_name, index=False)
    if cat_var != []:
        cat_var_full = df[cat_var]
        cat_name = "{}/cat_var_full.csv".format(libname)
        cat_var_full.to_csv(cat_name, index=False)
    # count the frequency of string variables and calculate the statistics of numerical variables
    fre_var = str_var + cat_var
    vari_df = []
    Level = df.apply(lambda x: len(x.unique()), axis=0)
    if fre_var != []:
        Proc_Freq_table = Proc_Freq(fre_var, n, df, logger)
        for vari in Proc_Freq_table.index.get_level_values(0).unique():
            vari_df.append([vari, Level[vari], Proc_Freq_table.xs(vari, level=0).to_html(classes='proc')])
    Proc_Uni_table = Proc_Uni(num_var, df, logger)
    len_Proc_Uni_table = len(Proc_Uni_table)
    Proc_Uni_table = Proc_Uni_table.to_html(classes='proc')
    return vari_df, Proc_Uni_table, Proc_contents, df.head(20), len_Proc_Uni_table, MissingRec, df_shape


####### HTML QC Report ################
def app(libname, filename, n):

    app = Flask(__name__)

    @app.route('/')
    def index():
        out1 = QCreport(libname, filename, n)
        return render_template('template.html', Str_Var=out1[0], title='Report', Proc_Uni=out1[1], file_name=filename, Proc_contents=out1[2], df=out1[3].to_html(classes='proc'), len_uni=out1[4],Missing=out1[5].to_html(classes='proc'), ncol=out1[6][1], nrow=out1[6][0])

    @app.template_global('current_time')
    def current_time(timeFormat="%b %d, %Y - %H:%M:%S"):
        return time.strftime(timeFormat)

    if __name__ == 'myfun':
        app.run(debug=True)
    return


######## Impute missing #########
def ImputeMissing(libname, filename, impute_method ='mean'):
    logger = mylog('ImputeMissing')
    file = "{}/{}.csv".format(libname, filename)
    temp_df = pd.read_csv(file, encoding='utf-8')
    imr = Imputer(missing_values='NaN', strategy=impute_method, axis=0)
    logger.info('"%s" strategy is used for imputing missing data' %(impute_method))
    imputed_df = pd.DataFrame(imr.fit_transform(temp_df))
    imputed_df.columns = temp_df.columns
    imputed_df.index = temp_df.index
    imputed_df.to_csv(file,index=False)
    return


######## CapFile ##############
def CapFile(libname, str_revised_in, num_revised_in, str_cap_out, num_cap_out, depend_var, level_n = 10):
    logger = mylog('CapFile')
    y_name = "{}/{}.csv".format(libname, depend_var)
    y_var = pd.read_csv(y_name, encoding='utf-8', header=None)[0].tolist()
    str_name = "{}/{}.csv".format(libname, str_revised_in)
    num_name = "{}/{}.csv".format(libname, num_revised_in)
    num_file = pd.read_csv(num_name, encoding='utf-8')
    num_file = pd.DataFrame(num_file)
    num_var = num_file.columns.tolist()
    num_var = [i for i in num_var if i not in y_var]
    if os.path.exists(str_name) == True:
        str_file = pd.read_csv(str_name, encoding='utf-8')
        str_file = pd.DataFrame(str_file)
        str_var = str_file.columns.tolist()
        str_var = [i for i in str_var if i not in y_var]
    # generate the capping rules for string variables
    CapFile = pd.DataFrame(columns=['varName', 'startValue', 'endValue', 'cappedValue'])
    if str_var != []:
        for i in str_var:
            logger.info('The capping rule of the "%s" string variable is being generated' % (i))
            level = len(str_file[i].unique().tolist())
            if level == 1:
                CapFile = CapFile.append(pd.DataFrame({'varName': i, 'startValue': str_file[i].unique(), 'cappedValue': 1}),
                                     ignore_index=True)
            elif level > 1 and level <= level_n:
                if len(str_file[i].isnull().unique()) == 2:
                # handle missing values
                    CapFile = CapFile.append(pd.DataFrame(
                    {'varName': i, 'startValue': np.append('missing', str_file[i].dropna().unique()),
                     'cappedValue': np.append(99, range(1, level))}), ignore_index=True)
                else:
                    CapFile = CapFile.append(pd.DataFrame(
                    {'varName': i, 'startValue': str_file[i].unique(), 'cappedValue': range(1, level + 1)}),
                                         ignore_index=True)
        Str_Cap_name = '{}/{}.csv'.format(libname, str_cap_out)
        CapFile.to_csv(Str_Cap_name, index=False)
    # generate the capping rules for numeriacal variables
    CapFileNum = pd.DataFrame(columns=['varName', 'startValue', 'endValue', 'cappedValue'])
    for n in num_var:
        logger.info('The capping rule of the "%s" numerical variable is being generated' % (n))
        quant = []
        for m in range(0, 11):
            quant.append(num_file[n].quantile(m / 10))
        new_quant = []
        for i in quant:
            if i not in new_quant:
                new_quant.append(i)
        level = len(new_quant)
        for i in range(0, level - 1):
            CapFileNum = CapFileNum.append(
                {'varName': n, 'startValue': new_quant[i], 'endValue': new_quant[(i + 1)], 'cappedValue': (i + 1)},
                ignore_index=True)
    Num_Cap_name = '{}/{}.csv'.format(libname, num_cap_out)
    CapFileNum.to_csv(Num_Cap_name, index=False)
    return


######## Merge CapFile with DataFile ##############
def docapping(libname, str_revised_in, num_revised_in, cat_revised_in, str_cap_in, num_cap_in, CappedData_out):
    logger = mylog('docapping')
    num_revised = pd.read_csv('{}/{}.csv'.format(libname, num_revised_in), encoding='utf-8')
    CapFileNum = pd.read_csv('{}/{}.csv'.format(libname, num_cap_in), encoding='utf-8')
    num_revised['key'] = num_revised.index.copy()
    # convert the numerical variables to the category variables
    CN_var = CapFileNum['varName'].unique().tolist()
    for i in CN_var:
        tmp = CapFileNum[CapFileNum['varName'] == i]
        tmp = tmp.reset_index()
        level = len(tmp)
        logger.info('The capping operation for numerical variable "%s" is being executed' % (i))
        temp_df = num_revised.sort_values(by = i)
        temp_df = temp_df.reset_index(drop = True)
        m = 0
        a0_value = tmp.loc[m, 'startValue']
        a0 = temp_df[temp_df[i] <= a0_value].index
        inx_temp = [temp_df.loc[a0, 'key'].values]
        while m < level:
            a_value = tmp.loc[m, 'startValue']
            b_value = tmp.loc[m, 'endValue']
            a = temp_df[temp_df[i] > a_value].index
            b = temp_df[temp_df[i] <= b_value].index
            inx = [miu for miu in a if miu in b]
            inx_temp.append(temp_df.loc[inx,'key'].values)
            m += 1
        num_revised.loc[inx_temp[0], i]= tmp['cappedValue'][0]
        m = 0
        while m < level:
            num_revised.loc[inx_temp[m + 1], i] = tmp['cappedValue'][m]
            m += 1
    # convert the string variables to the category variables
    if os.path.exists('{}/{}.csv'.format(libname, str_revised_in)) == True:
        str_revised = pd.read_csv('{}/{}.csv'.format(libname, str_revised_in), encoding='utf-8')
        CapFileStr = pd.read_csv('{}/{}.csv'.format(libname, str_cap_in), encoding='utf-8')
        str_revised['key'] = str_revised.index.copy()
        CS_var = CapFileStr['varName'].unique().tolist()
        for i in CS_var:
            tmp = CapFileStr[CapFileStr['varName'] == i]
            tmp = tmp.reset_index()
            tmp = tmp.drop(['varName', 'endValue'], axis = 1)
            str_revised[i] = str_revised[i].fillna('missing')
            logger.info('The capping operation for string variable "%s" is being executed' % (i))
            str_revised = pd.merge(str_revised, tmp, how='outer', left_on=i, right_on='startValue')
            str_revised = str_revised.sort_values(by='key')
            str_revised = str_revised.drop(['startValue', i, 'index'], axis = 1)
            str_revised = str_revised.rename(columns={'cappedValue': i})
        CappedDF = pd.merge(str_revised, num_revised, on='key', how='outer')
    else:
        CappedDF = num_revised
    # add the category variables if they exist
    if os.path.exists('{}/{}.csv'.format(libname, cat_revised_in)) == True:
        cat_revised = pd.read_csv('{}/{}.csv'.format(libname, cat_revised_in), encoding='utf-8')
        cat_revised['key'] = cat_revised.index.copy()
        CappedDF = pd.merge(CappedDF, cat_revised, on='key', how='outer')
    CappedDF = CappedDF.drop('key', axis=1)
    CappedFile_name = '{}/{}.csv'.format(libname, CappedData_out)
    CappedDF.to_csv(CappedFile_name, index=False)
    return


######## Regression Between single x & y ###########
def doLM(libname, DF, y_in):
    logger = mylog('doLM')
    df = pd.read_csv('{}/{}.csv'.format(libname, DF), encoding='utf-8')
    response = pd.read_csv('{}/{}.csv'.format(libname, y_in), encoding='utf-8',header = None)
    allvar = df.columns.tolist()
    lm_out = pd.DataFrame(columns = ['y','x','Coeff','f_test','p_value'])
    for i in range(len(response)):
        # get the independent y variable
        y = response.iloc[i].dropna().tolist()
        df = df.dropna(subset = y)
        # get the dependent x variables
        x_all = list(set(allvar).difference(set(y)))
        obj_var = df.select_dtypes(include=['object']).columns.tolist()
        x_all = [item for item in x_all if item not in response[0].tolist() and item not in obj_var]
        for j in range(len(x_all)):
            # fit the OLS model
            logger.info('The regression between single "%s" and "%s" is being executed' % (x_all[j], response[0].tolist()[i]))
            y_df = df.filter(items = y)
            x = x_all[j]
            x_df = df.filter(items = [x])
            x_df = x_df.fillna(np.mean(x_df))
            x_df = sm.add_constant(x_df,prepend=False)
            model = sm.OLS(y_df, x_df).fit()
            temp = pd.DataFrame({'y':str(y[0]), 'x':x, 'Coeff': [model.params.values[0]],'f_test': [model.fvalue],'p_value': [model.f_pvalue]})
            lm_out = lm_out.append(temp)
    lm_out.to_csv('{}/doLM.csv'.format(libname), index = False)


######## Correlation Between Variables ##############
def Corr(libname, CappedData_in, corr_out, y_varname, cutoff=0.5):
    logger = mylog('Corr')
    filename = '{}/{}.csv'.format(libname, CappedData_in)
    y_name = "{}/{}.csv".format(libname, y_varname)
    y_var = pd.read_csv(y_name, encoding='utf-8', header=None)[0].tolist()
    CappedDF = pd.read_csv(filename, encoding='utf-8')
    CappedDF = CappedDF.drop(y_var, axis = 1)
    Corr_df = CappedDF.select_dtypes(exclude=['object'])
    # calculate the pearson correlation between variables
    corr = Corr_df.corr().dropna(axis=1, how='all')
    corr = corr.dropna(axis=0, how='all')
    corr_triu = corr.where(~np.tril(np.ones(corr.shape)).astype(np.bool))
    corr_triu = corr_triu.stack()
    logger.info('The variables between which correlation exceeds the threshold is being selected')
    # select variables whose correlation is larger than cutoff
    corr_list = corr_triu[abs(corr_triu) > cutoff]
    corr_list = corr_list.sort_values(ascending=False)
    corr_list = corr_list.reset_index()
    corr_list = corr_list.rename(columns={'level_0': 'Var1', 'level_1': 'Var2', 0: 'Pearson_Corr'})
    length = len(corr_list.index)
    corr_list['cutOff_Lower_limit'] = [cutoff] * length
    corr_list['absCorr'] = abs(corr_list['Pearson_Corr'])
    corr_out_name = '{}/{}.csv'.format(libname, corr_out)
    corr_list.to_csv(corr_out_name, index=False)
    return


######## PCA datafile/NA/cutoff ##############
def pre_pca(libname, corr_in, CappedData_in, corr_thresh_out, pre_pca_out, thresh=0.75, method='mean'):
    corr_in_name = '{}/{}.csv'.format(libname, corr_in)
    corr_in = pd.read_csv(corr_in_name, encoding='utf-8')
    filename = '{}/{}.csv'.format(libname, CappedData_in)
    CappedDF = pd.read_csv(filename, encoding='utf-8')
    # select variables whose correlation is less than thresh
    corr_thresh = corr_in[corr_in['absCorr'] <= thresh]
    corr_out_name = '{}/{}.csv'.format(libname, corr_thresh_out)
    corr_thresh.to_csv(corr_out_name, index=False)
    var = corr_thresh['Var1'].append(corr_thresh['Var2']).unique().tolist()
    PCA_df = CappedDF.filter(items=var)
    # fill in the missing values
    if method is 'mean':
        PCA_df = PCA_df.apply(lambda x: x.fillna((np.mean(x)).round(2)))
    elif method is 'median':
        PCA_df = PCA_df.apply(lambda x: x.fillna((np.median(x)).round(2)))
    else:
        PCA_df = PCA_df.apply(lambda x: x.fillna((np.mean(x)).round(2)))
    inx = pd.DataFrame(CappedDF.iloc[:, 0])
    PCA_df = pd.concat([inx, PCA_df], axis=1)
    out_file_name = '{}/{}.csv'.format(libname, pre_pca_out)
    PCA_df.to_csv(out_file_name, index=False)
    return


######## PCA & PCA Report ##############
def pca(libname, CappedData_in, CappedData_out, PCA_df, PCAGroup, new_var_out, PCAmethod=None, scale='True'):
    logger = mylog('PCA')
    df_name = "{}/{}.csv".format(libname, CappedData_in)
    df = pd.read_csv(df_name, encoding='utf-8')
    pca_name = "{}/{}.csv".format(libname, PCA_df)
    pca_df = pd.read_csv(pca_name, encoding='utf-8')
    filename = "{}/{}.csv".format(libname, PCAGroup)
    group = pd.read_csv(filename, encoding='utf-8', header=None)
    group_var = []
    for i in range(0, group.shape[0]):
        group_var = group_var + group.iloc[i].dropna().tolist()
    group_var = list(set(group_var))
    length = len(group)
    Simple_Stat = (pca_df.describe()).loc[['mean', 'std']]
    vari_d = []
    equation = []
    new_var = pd.DataFrame()
    Summ_out = pd.DataFrame()
    logger.info('The PCA is being carried out')
    for i in range(0, length):
        var = group.loc[i,:].dropna().tolist()
        tmp = pca_df.filter(items=var)
        x = tmp.values
        # data standardization
        if scale is 'True':
            x = StandardScaler().fit_transform(x)
        elif scale is 'False':
            pass
        Simple_Stat_tb = Simple_Stat.filter(items=var)
        Cov_Matrix = tmp.corr()
        n_comp = len(var)
        # PCA
        pca = PCA(n_components=n_comp)
        pca.fit(x)
        EigenvalCor = pd.DataFrame(columns=['Eigenvalue', 'Proportion', 'Cumulative'])
        # report eigenvalue
        EigenvalCor['Eigenvalue'] = pca.explained_variance_
        # report % variance
        eigenvalue = pca.explained_variance_ratio_
        EigenvalCor['Proportion'] = eigenvalue
        # report cumulative variance
        cu_eigenvalue_var = np.cumsum(np.round(eigenvalue, decimals=2))
        EigenvalCor['Cumulative'] = cu_eigenvalue_var
        # report eigenvector
        eigenvector = pca.components_
        eigenvector = eigenvector.transpose()
        col_comp = colname_suffix(n_comp, 'comp')
        Eigenvectors = pd.DataFrame(data=eigenvector, columns=col_comp, index=var)
        mean = np.mean(tmp).round(2).tolist()
        std = np.std(tmp).round(2).tolist()
        Summary = pd.DataFrame({'VarName': var, 'Mean': mean, 'std': std, 'Eigenval': eigenvalue})
        Summary = pd.merge(Summary, Eigenvectors, left_on='VarName', right_index=True)
        out = []
        # report equation
        for k in range(0, n_comp):
            a = '{} = '.format(col_comp[k])
            q = 0
            while q < n_comp:
                b = '({})*({}-{})/{}'.format(eigenvector[q, k].round(2), var[q], mean[q], std[q])
                a = '{}+{}'.format(a, b)
                q += 1
            a = a.replace("+", "", 1)
            out.append(a)
        equation = pd.DataFrame(columns=['Equation'], data=out)
        Var_name = 'New_Var{}'.format(i + 1)
        Var_tot = '/'.join(var)
        vari_d.append([len(tmp), n_comp, ('{}={}'.format(Var_name, Var_tot)), Simple_Stat_tb.to_html(classes='proc'),
                       Cov_Matrix.to_html(classes='proc'), EigenvalCor.to_html(classes='proc'),
                       Eigenvectors.to_html(classes='proc'), Summary.to_html(classes='proc'),
                       equation.to_html(classes='equ', border=None, index=False)])
        Summ_out = pd.concat([Summ_out, Summary], ignore_index=False,sort=False)
        # use the mle method for PCA
        if PCAmethod is 'mle':
            pca_mle = PCA(n_components='mle', svd_solver='full')
            pca_mle.fit(x)
            dim = len(pca_mle.components_)
            new_x = pca_mle.fit_transform(x).tolist()
        # no dimensionality reduction
        elif PCAmethod is None:
            dim = n_comp
            new_x = pca.fit_transform(x).tolist()
        # reduce to a specified dimension
        else:
            if PCAmethod > n_comp:
                method_tmp = n_comp
            else:
                method_tmp = PCAmethod
            pca_n = PCA(n_components=method_tmp)
            pca_n.fit(x)
            dim = len(pca_n.components_)
            new_x = pca_n.fit_transform(x).tolist()
        colname = colname_suffix(dim, 'New_Var{}_'.format(i + 1))
        tmp_df = pd.DataFrame(columns=colname, data=new_x, index=range(0, len(x)))
        new_var = pd.concat([new_var, tmp_df], axis=1,sort=False)
    inx = pd.DataFrame(pca_df.iloc[:, 0])
    new_var = pd.concat([inx, new_var], axis=1,sort=False)
    if new_var.empty == False:
        new_var.to_csv("{}/{}.csv".format(libname, new_var_out), index=False)
    df = df.drop(group_var, axis = 1)
    df = pd.merge(df, new_var, on = df.columns.tolist()[0])
    df.to_csv("{}/{}.csv".format(libname, CappedData_out), index=False)
    Summ_out.to_csv("{}/PCA_summary.csv".format(libname))
    return vari_d, new_var


def app2(libname, CappedData_in, CappedData_out, PCA_df, PCAGroup, new_var_out, PCAmethod=None, scale='True'):
    app = Flask(__name__)

    @app.route('/')
    def index():
        out2 = pca(libname, CappedData_in, CappedData_out, PCA_df, PCAGroup, new_var_out, PCAmethod, scale)
        return render_template('template2.html', Var=out2[0], title='PCA Report')

    @app.template_global('current_time')
    def current_time(timeFormat="%b %d, %Y - %H:%M:%S"):
        return time.strftime(timeFormat)

    if __name__ == 'myfun':
        app.run(debug=True)
    return


############### Delete Duplicates #################
def dedup(stringw):
    # delete duplicate characters in a string
    string_arr = stringw.split()
    _unique__ = []
    for item in string_arr:
        if item not in _unique__:
            _unique__.append(item)
    return _unique__


############### Attach Splits #################
def attachsplits(lists, splits):
    # get the cross subset between split variables and univariates
    myList = []
    List = splits + ' ' + lists
    List_arr = List.split()
    splits_arr = splits.split()
    for i in splits_arr:
        for j in List_arr:
            if i != j:
                tmp_str = set(i.split('*')) & set(j.split('*'))
                if len(tmp_str) == 0:
                    myList.append(i + '*' + j)
                else:
                    for k in tmp_str:
                        j = j.split('*')
                        j.remove(k)
                    myList.append(i + '*' + '*'.join(j))
    outList = myList
    outClassVars = dedup(List.replace('*',' '))
    return outList, outClassVars


############### Reset the value of level #################
def reset_level(temp, cap_str_var, cap_num_var, cap_str, cap_num):
    name_dict = []
    for index_name in temp.index.names:
        # match capping rules of string varibles
        if index_name in cap_str_var:
            dict_temp = cap_str[cap_str['varName'] == index_name]
            dict_list= list(zip(dict_temp['cappedValue'].values.transpose(),dict_temp['startValue'].values.transpose()))
            dict_list_new = ['({}):{}'.format(str(int(i[0])).zfill(2), i[1]) for i in dict_list]
            str_dict= dict(zip(dict_temp['cappedValue'].values.transpose(), dict_list_new))
            name_dict.append(str_dict)
        # match capping rules of numerical varibles
        elif index_name in cap_num_var:
            dict_temp = cap_num[cap_num['varName'] == index_name]
            dict_list = dict_temp[['startValue', 'endValue', 'cappedValue']].values.tolist()
            dict_list_new = ['({}):{}-{}'.format(str(int(i[2])).zfill(2), '{:.1f}'.format(i[0]), '{:.1f}'.format(i[1])) for i in dict_list]
            num_dict= dict(zip(dict_temp['cappedValue'].values.transpose(), dict_list_new))
            if 0.0 in temp.index.get_level_values(index_name) or 0 in temp.index.get_level_values(index_name):
                try:
                    num_dict[0.0] = '(00):missing'
                except:
                    num_dict[0] = '(00):missing'
            name_dict.append(num_dict)
        # format the category varibles
        else:
            cat_inx = list(set(temp.index.get_level_values(index_name)))
            cat_val = ['({}):'.format(str(int(i)).zfill(2)) for i in cat_inx]
            cat_dict = dict(zip(tuple(cat_inx), cat_val))
            name_dict.append(cat_dict)
    return name_dict


############### Analysis of Interaction between Variables #################
def fastuni2(libname, filename, varname, capnum, capstr):
    # define the logger
    logger = mylog('fastuni2')
    # load the capping rules for string and numerical variables
    cap_num = pd.read_csv("{}/{}.csv".format(libname, capnum), encoding='utf-8')
    cap_str = pd.read_csv("{}/{}.csv".format(libname, capstr), encoding='utf-8')
    cap_num_var = cap_num['varName'].unique().tolist()
    cap_str_var = cap_str['varName'].unique().tolist()
    # load the dataset and get its variables
    file = "{}/{}.csv".format(libname, filename)
    df = pd.read_csv(file, encoding='utf-8')
    vars = df.columns.tolist()
    # load input variables involved in the univariate analysis
    varfile = "{}/{}.csv".format(libname, varname)
    variable = pd.read_csv(varfile, encoding='utf-8')
    variable = pd.DataFrame(variable).where(variable.notnull(), None)
    where = ' '.join(variable['where'].replace([None], '').tolist())
    uni_list = ' '.join(variable['list'].replace([None], '').tolist())
    splits = ' '.join(variable['splits'].replace([None], '').tolist())
    ratioList = ' '.join(variable['ratioList'].replace([None], '').tolist())
    pctvars = ' '.join(variable['pctvars'].replace([None], '').tolist())
    relBaseLevel = ' '.join(variable['relBaseLevel'].replace([None], '').astype('str').tolist())
    statVars = ' '.join(variable['statVars'].replace([None], '').tolist())
    statsWgt = ' '.join(variable['statsWgt'].replace([None], '').tolist())
    stats = ' '.join(variable['stats'].replace([None], '').tolist())
    if stats.strip() == '':
        stats = 'mean'
    keep = ' '.join(variable['keep'].replace([None], '').tolist())
    # match input variables with variables in the dataset
    logger.info('Match variables')
    if where.strip() != '':
        addl_keep = [i for i in vars if where.find(i) != -1]
    else:
        addl_keep = []
    if statVars.strip() != '':
        statVars = [i for i in statVars.split() if i in vars]
    else:
        statVars = []
    if statsWgt.strip() != '':
        statsWgt = [i for i in statsWgt.split() if i in vars]
    else:
        statsWgt = []
    stats = stats.split()
    if uni_list.strip() != '':
        uni_list = [i for i in uni_list.split() if set(re.split('[ *]', i)).issubset(set(vars))]
    else:
        uni_list = []
    if pctvars.strip() != '':
        pctvars = [i for i in pctvars.split() if i in vars]
    else:
        pctvars = []
    # Separate ratiodList to 3 separate lists
    ratio = []
    relativity = []
    numerator = []
    denom = []
    if ratioList.strip() != '':
        ratioList_split = ratioList.split()
        numOfVars = len(ratioList_split)
        for i in range(0, numOfVars, 3):
            if ratioList_split[i + 1] in vars and ratioList_split[i + 2] in vars:
                ratio.append(ratioList_split[i])
                relativity.append(ratioList_split[i] + '_REL')
                numerator.append(ratioList_split[i + 1])
                denom.append(ratioList_split[i + 2])
    sumvars = dedup(' '.join(denom) + ' ' + ' '.join(numerator) + ' ' + ' '.join(pctvars))
    if splits.strip() != '':
        splits = [i for i in splits.split() if set(re.split('[ *]', i)).issubset(set(vars))]
        minilist, classvars = attachsplits(lists=' '.join(uni_list), splits=' '.join(splits))
        if relBaseLevel.strip() == '':
            relBaseLevel = 1
    else:
        splits = []
        minilist = dedup(' '.join(uni_list))
        classvars = dedup(' '.join(uni_list))
        if relBaseLevel.strip() == '':
            relBaseLevel = 0
    # determine variables that must be kept for the where clause
    query_condi = sumvars + addl_keep + classvars + statVars + statsWgt
    if where.strip() == '':
        df1 = df[query_condi]
        df1 = df1.loc[:, ~df1.columns.duplicated()].fillna(0)
    else:
        try:
            df1 = df.query(where)[query_condi]
            df1 = df1.loc[:, ~df1.columns.duplicated()].fillna(0)
        except:
            logger.exception(sys.exc_info())
            logger.info('Check whether variables in the "where" clause are in the dataset')

    stat_statVars = []
    # group summation of sumvars by the cross subset in the minilist
    # reset the value of index
    out_1 = pd.DataFrame()
    for cross_var in minilist:
        group_var = cross_var.split('*')
        logger.info('The grouped sum value of "pctvars" and "ratioList" of "%s" are being calculated' %cross_var)
        temp = df1.groupby(group_var)[sumvars].sum().add_prefix('sum_')
        name_dict = reset_level(temp, cap_str_var, cap_num_var, cap_str, cap_num)
        for i in range(0, len(group_var)):
            temp = temp.rename(dict(name_dict[i]), level=i, axis='index')
        temp = temp.reset_index(group_var)
        out_1 = pd.concat([out_1, temp], axis=0, ignore_index=True, sort=False)
    # group statistics of statVars by the cross subset in the minilist
    # reset the value of index
    out_1_1 = pd.DataFrame()
    if statVars != [] and statsWgt == []:
        for cross_var in minilist:
            group_var = cross_var.split('*')
            logger.info('The value of "group_var" of "%s" is resetted' %cross_var)
            temp = df1.groupby(group_var)[statVars].mean().add_prefix('mean_')
            name_dict = reset_level(temp, cap_str_var, cap_num_var, cap_str, cap_num)
            for i in range(0, len(group_var)):
                temp = temp.rename(dict(name_dict[i]), level=i, axis='index')
            temp = temp.reset_index(group_var)[group_var]
            logger.info('The grouped statistics of "statVars" of "%s" are being calculated' %cross_var)
            for stat_item in stats:
                if stat_item == 'mean':
                    temp_1 = df1.groupby(group_var)[statVars].mean().add_prefix('mean_')
                    stat_statVars += temp_1.columns.tolist()
                    name_dict = reset_level(temp_1, cap_str_var, cap_num_var, cap_str, cap_num)
                    for i in range(0, len(group_var)):
                        temp_1 = temp_1.rename(dict(name_dict[i]), level=i, axis='index')
                    temp_1 = temp_1.reset_index(group_var)
                    temp = pd.merge(temp, temp_1, how='outer', on=group_var)
                if stat_item == 'sum':
                    temp_1 = df1.groupby(group_var)[statVars].sum().add_prefix('sum_')
                    stat_statVars += temp_1.columns.tolist()
                    name_dict = reset_level(temp_1, cap_str_var, cap_num_var, cap_str, cap_num)
                    for i in range(0, len(group_var)):
                        temp_1 = temp_1.rename(dict(name_dict[i]), level=i, axis='index')
                    temp_1 = temp_1.reset_index(group_var)
                    temp = pd.merge(temp, temp_1, how='outer', on=group_var)
                if stat_item == 'median':
                    temp_1 = df1.groupby(group_var)[statVars].median().add_prefix('median_')
                    stat_statVars += temp_1.columns.tolist()
                    name_dict = reset_level(temp_1, cap_str_var, cap_num_var, cap_str, cap_num)
                    for i in range(0, len(group_var)):
                        temp_1 = temp_1.rename(dict(name_dict[i]), level=i, axis='index')
                    temp_1 = temp_1.reset_index(group_var)
                    temp = pd.merge(temp, temp_1, how='outer', on=group_var)
                if stat_item == 'var':
                    temp_1 = df1.groupby(group_var)[statVars].var().add_prefix('var_')
                    stat_statVars += temp_1.columns.tolist()
                    name_dict = reset_level(temp_1, cap_str_var, cap_num_var, cap_str, cap_num)
                    for i in range(0, len(group_var)):
                        temp_1 = temp_1.rename(dict(name_dict[i]), level=i, axis='index')
                    temp_1 = temp_1.reset_index(group_var)
                    temp = pd.merge(temp, temp_1, how='outer', on=group_var)
                if stat_item == 'std':
                    temp_1 = df1.groupby(group_var)[statVars].std().add_prefix('std_')
                    stat_statVars += temp_1.columns.tolist()
                    name_dict = reset_level(temp_1, cap_str_var, cap_num_var, cap_str, cap_num)
                    for i in range(0, len(group_var)):
                        temp_1 = temp_1.rename(dict(name_dict[i]), level=i, axis='index')
                    temp_1 = temp_1.reset_index(group_var)
                    temp = pd.merge(temp, temp_1, how='outer', on=group_var)
            out_1_1 = pd.concat([out_1_1, temp], axis=0, ignore_index=True, sort=False)
    # group weighted statistics of statVars by the cross subset in the minilist
    # reset the value of index
    out_1_2 = pd.DataFrame()
    if statVars != [] and statsWgt != []:
        for cross_var in minilist:
            group_var = cross_var.split('*')
            logger.info('The value of "group_var" of "%s" is resetted' %cross_var)
            temp = df1.groupby(group_var)[statVars].mean().add_prefix('mean_')
            name_dict = reset_level(temp, cap_str_var, cap_num_var, cap_str, cap_num)
            for i in range(0, len(group_var)):
                temp = temp.rename(dict(name_dict[i]), level=i, axis='index')
            temp = temp.reset_index(group_var)[group_var]
            logger.info('The grouped weighted statistics of "statVars" of "%s" are being calculated' %cross_var)
            for stat_item in stats:
                if stat_item == 'mean':
                    wm = lambda x: np.average(x, weights=df1.loc[x.index, statsWgt[0]])
                    temp_2 = df1.groupby(group_var)[statVars].agg(wm).add_prefix('mean_')
                    stat_statVars += temp_2.columns.tolist()
                    name_dict = reset_level(temp_2, cap_str_var, cap_num_var, cap_str, cap_num)
                    for i in range(0, len(group_var)):
                        temp_2 = temp_2.rename(dict(name_dict[i]), level=i, axis='index')
                    temp_2 = temp_2.reset_index(group_var)
                    temp = pd.merge(temp, temp_2, how='outer', on=group_var)
                if stat_item == 'sum':
                    ws = lambda x: np.sum(x * df1.loc[x.index, statsWgt[0]])
                    temp_2 = df1.groupby(group_var)[statVars].agg(ws).add_prefix('sum_')
                    stat_statVars += temp_2.columns.tolist()
                    name_dict = reset_level(temp_2, cap_str_var, cap_num_var, cap_str, cap_num)
                    for i in range(0, len(group_var)):
                        temp_2 = temp_2.rename(dict(name_dict[i]), level=i, axis='index')
                    temp_2 = temp_2.reset_index(group_var)
                    temp = pd.merge(temp, temp_2, how='outer', on=group_var)
                if stat_item == 'median':
                    wm = lambda x: np.median(x * df1.loc[x.index, statsWgt[0]])
                    temp_2 = df1.groupby(group_var)[statVars].agg(wm).add_prefix('median_')
                    stat_statVars += temp_2.columns.tolist()
                    name_dict = reset_level(temp_2, cap_str_var, cap_num_var, cap_str, cap_num)
                    for i in range(0, len(group_var)):
                        temp_2 = temp_2.rename(dict(name_dict[i]), level=i, axis='index')
                    temp_2 = temp_2.reset_index(group_var)
                    temp = pd.merge(temp, temp_2, how='outer', on=group_var)
                if stat_item == 'var':
                    wv = lambda x: (df1.loc[x.index, statsWgt[0]] * (x - np.mean(x)) ** 2).sum() / df1.loc[x.index, statsWgt[0]].sum()
                    temp_2 = df1.groupby(group_var)[statVars].agg(wv).add_prefix('var_')
                    stat_statVars += temp_2.columns.tolist()
                    name_dict = reset_level(temp_2, cap_str_var, cap_num_var, cap_str, cap_num)
                    for i in range(0, len(group_var)):
                        temp_2 = temp_2.rename(dict(name_dict[i]), level=i, axis='index')
                    temp_2 = temp_2.reset_index(group_var)
                    temp = pd.merge(temp, temp_2, how='outer', on=group_var)
                if stat_item == 'std':
                    ws = lambda x: np.sqrt((df1.loc[x.index, statsWgt[0]] * (x - np.mean(x)) ** 2).sum() / df1.loc[x.index, statsWgt[0]].sum())
                    temp_2 = df1.groupby(group_var)[statVars].agg(ws).add_prefix('std_')
                    stat_statVars += temp_2.columns.tolist()
                    name_dict = reset_level(temp_2, cap_str_var, cap_num_var, cap_str, cap_num)
                    for i in range(0, len(group_var)):
                        temp_2 = temp_2.rename(dict(name_dict[i]), level=i, axis='index')
                    temp_2 = temp_2.reset_index(group_var)
                    temp = pd.merge(temp, temp_2, how='outer', on=group_var)
            out_1_2 = pd.concat([out_1_2, temp], axis=0, ignore_index=True, sort=False)
    if out_1_1.empty == True and out_1_2.empty == False:
        out = pd.merge(out_1, out_1_2, how='outer', on=classvars)
    elif out_1_1.empty == False and out_1_2.empty == True:
        out = pd.merge(out_1, out_1_1, how='outer', on=classvars)
    else:
        out = out_1
    out = out.drop_duplicates().reset_index(drop=True)
    stat_statVars = list(set(stat_statVars))

    # get the maximum number of cross subsets
    maxway = 0
    for item in minilist:
        max = len(item.split('*'))
        if maxway < max:
            maxway = max
    # determin split and level based on grouping variables
    # split corresponds to the variable name and level corresponds to the variable value before capping
    Split = []
    Level = []
    for i in range(1, maxway + 1):
        out['split' + str(i)] = 'ALL'
        Split.append('split' + str(i))
        out['level' + str(i)] = 'ALL'
        Level.append('level' + str(i))
    ini_len = out.shape[0]
    logger.info('The value of split and level is being matched')
    for i in range(0, ini_len):
        tmp = out.loc[i]
        tmp_split = tmp[classvars]
        inx = tmp_split[~tmp_split.isnull()].index.tolist()
        out.loc[i, Split[:len(inx)]] = inx
        out.loc[i, Level[:len(inx)]] = [tmp[j] for j in inx]
        # special case for when running univariates;
        # the summarized dataset has only two split levels;
        if inx[0] in splits and inx[1] in splits:
            out = out.append(out.loc[i], ignore_index=True)
            upVarName = inx[0]
            varValue = tmp[inx[0]]
            out.loc[out.shape[0] - 1, 'split1'] = inx[1]
            out.loc[out.shape[0] - 1, 'level1'] = tmp[inx[1]]
            out.loc[out.shape[0] - 1, 'split2'] = upVarName
            out.loc[out.shape[0] - 1, 'level2'] = varValue
    out = out.drop(classvars, axis=1).sort_values(by=Split + Level).drop_duplicates().reset_index(drop=True)
    # determin the split_all based on the relBaseLevel and maxway
    # calculate the overall sum for sumvars and the overall statistics for statVars
    out_all = pd.DataFrame(columns=out.columns.tolist())
    for item in sumvars:
        out_all.loc[0, 'sum_' + item] = np.sum(df1[item])
    final_output = pd.DataFrame(columns=Split + Level)
    relBaseLevel = int(float(relBaseLevel))
    if relBaseLevel > maxway:
        relBaseLevel = maxway
    sum_sumvars = ['sum_' + item for item in sumvars]
    # get the fixed interaction variables
    all_var = out[Split].drop_duplicates().values.tolist()
    split_fix = []
    level_fix = []
    for i in range(1, relBaseLevel + 1):
        split_fix.append('split' + str(i))
        level_fix.append('level' + str(i))
    # calculate the ratio, relativity, and percent of pctvars
    logger.info('The ratio, relativities, and percent are being calculated')
    if split_fix != [] and relBaseLevel < maxway:
        for item in all_var:
            inter_var = item[relBaseLevel:]
            inx = relBaseLevel + 1
            for uni_var in inter_var:
                if uni_var != 'ALL':
                    # count the frequency
                    count_1 = 0
                    for i in all_var:
                        if i[inx - 1] == uni_var:
                            count_1 += 1
                    dup_list = [tuple(i[0: relBaseLevel]) for i in all_var if i[inx - 1] == uni_var]
                    dup_list = Counter(dup_list)
                    # in the case of "ALL"
                    tmp_df_1 = out.fillna(0)[out['split' + str(inx)] == uni_var].reset_index(drop=True)
                    tmp_df_1[sum_sumvars] = tmp_df_1[sum_sumvars] / count_1
                    tmp_df_1 = tmp_df_1.groupby(by=['split' + str(inx), 'level' + str(inx)])[sum_sumvars].sum().reset_index()
                    # in the case of "SPLIT"
                    tmp_df_2 = out.fillna(0)[out['split' + str(inx)] == uni_var].reset_index(drop=True)
                    tmp_df_2 = tmp_df_2.groupby(by=split_fix + level_fix + ['split' + str(inx)] + ['level' + str(inx)])[sum_sumvars].sum().reset_index()
                    # count the frequency
                    def count_func(x):
                        keys = tuple(x.values.tolist())
                        return dup_list[keys]
                    tmp_df_2['count'] = tmp_df_2.apply(lambda x: count_func(x[split_fix]), axis=1)
                    for i in sum_sumvars:
                        tmp_df_2[i] = tmp_df_2[i] / tmp_df_2['count']
                    tmp_all = tmp_df_2.groupby(by=split_fix + level_fix)[sum_sumvars].sum().reset_index()
                    # group summation
                    for i in sumvars:
                        tmp_val = dict(zip(tuple(map(tuple, tmp_all[split_fix + level_fix].values)),tmp_all['sum_' + i].values.tolist()))
                        def match_func(x):
                            keys = tuple(x.values.tolist())
                            return tmp_val[keys]
                        tmp_df_2[i + '_all'] = tmp_df_2.apply(lambda x: match_func(x[split_fix + level_fix]), axis=1)
                    # calculate the percent of pctvars
                    logger.info('The percent of "%s" is calculated' % (uni_var))
                    if pctvars != []:
                        pct = []
                        for i in pctvars:
                            pct.append('pct_' + i)
                            tmp_df_1['pct_' + i] = tmp_df_1['sum_' + i] / out_all['sum_' + i].values[0]
                            tmp_df_1['pct_' + i] = tmp_df_1['pct_' + i].apply(lambda x: '{}%'.format(round(x * 100, 2)))
                            tmp_df_2['pct_' + i] = tmp_df_2['sum_' + i] / tmp_df_2[i + '_all']
                            tmp_df_2['pct_' + i] = tmp_df_2['pct_' + i].apply(lambda x: '{}%'.format(round(x * 100, 2)))
                    # calculate the ratio and relativities of 'ratio'
                    logger.info('The ratio and relativities of "%s" is calculated' % (uni_var))
                    for i in range(0, len(ratio)):
                        ratio_word = ratio[i]
                        relativity_word = relativity[i]
                        numerator_word = numerator[i]
                        denom_word = denom[i]
                        tmp_df_1[ratio_word] = tmp_df_1['sum_' + numerator_word] / tmp_df_1['sum_' + denom_word]
                        rel_value = out_all['sum_' + denom_word].values[0] / out_all['sum_' + numerator_word].values[0]
                        tmp_df_1[relativity_word] = tmp_df_1[ratio_word] * rel_value - 1
                        tmp_df_1[ratio_word] = tmp_df_1[ratio_word].apply(lambda x: '{}%'.format(round(x * 100, 2)))
                        tmp_df_1[relativity_word] = tmp_df_1[relativity_word].apply(lambda x: '{}%'.format(round(x * 100, 2)))
                        tmp_df_2[ratio_word] = tmp_df_2['sum_' + numerator_word] / tmp_df_2['sum_' + denom_word]
                        tmp_df_2[relativity_word] = tmp_df_2[ratio_word] * tmp_df_2[denom_word + '_all'] / tmp_df_2[numerator_word + '_all'] - 1
                        tmp_df_2[ratio_word] = tmp_df_2[ratio_word].apply(lambda x: '{}%'.format(round(x * 100, 2)))
                        tmp_df_2[relativity_word] = tmp_df_2[relativity_word].apply(lambda x: '{}%'.format(round(x * 100, 2)))
                    tmp_df = pd.concat([tmp_df_1, tmp_df_2], axis=0, ignore_index=True)
                    final_output = pd.concat([final_output, tmp_df], axis=0, ignore_index=True, sort=False)
                inx += 1

    # detemine the final output
    logger.info('The files are being saved')
    final_output[Split + Level] = final_output[Split + Level].fillna('ALL')
    final_output = final_output.drop_duplicates()
    for item in sumvars:
        sum_item = 'sum_' + item
        final_output.rename(columns={sum_item: item}, inplace=True)
    if keep.strip() == '' and pctvars != []:
        keep_split = Split + Level + pctvars + pct + ratio + relativity
    if keep.strip() == '' and pctvars == []:
        keep_split = Split + Level + ratio + relativity
    if keep.strip() != '' and pctvars != []:
        keep_split = Split + Level + pct + ratio + relativity + keep.split()
    if keep.strip() != '' and pctvars == []:
        keep_split = Split + Level + ratio + relativity + keep.split()
    final_output = final_output[keep_split]
    out.to_csv("{}/{}.csv".format(libname, 'groupby_statistics'), index=False)
    final_output.to_csv("{}/{}.csv".format(libname, 'interaction_results'), index=False, columns=keep_split)
    return


############### Distribution #################
def dvdistribution(libname, CappedData_in, y_in):
    data = pd.read_csv("{}/{}.csv".format(libname, CappedData_in), encoding='utf-8')
    y_name = pd.read_csv("{}/{}.csv".format(libname, y_in), encoding='utf-8', header=None)
    response = list(chain(*y_name.values))
    for item in response:
        sns.set(palette="muted", color_codes=True)
        sns.distplot(data[item].dropna(), color="g")
        plt.xlabel(item)
        plt.ylabel('Density')
        plt.title('Distribution of %s' %item)
        figname = '{}/distribution_{}.png'.format(libname, item)
        plt.savefig(figname, dpi=100, bbox_inches='tight')
        plt.close('all')
    return


############### Unwind Parameters #################
def unwindparam(pre, group, summ, results,logger):
    # confirm whether there are pca variables
    new=[]
    round = 0
    index=[]
    for q in pre:
        round += 1
        if 'New_Var' in q:
            new.append(q)
            index.append(round)
    comp=[]
    var=[]
    # get pre- and post-dimensionality variables
    logger.info('The post-PCA variables are being restored')
    for i in new:
        x = [m.start() for m in re.finditer('_',i)][1]
        comp.append(int(i[x+1:]))
        var.append(int(i[7:x]))
    length = [len(group.iloc[x,:].dropna()) for x in range(len(group))]
    unwind=pd.DataFrame()
    inter= pd.DataFrame()
    for i in range(len(var)):
        var_id = var[i]
        s = 1
        index_in = 0
        while s < var_id:
            index_in = index_in +length[s-1]
            s +=1
        index_out = index_in + length[var_id-1]
        # restore the coefficient of the new dimension to coefficients of original dimensions
        df = summ.iloc[index_in:index_out,:].reset_index(drop = True)
        com = 'comp{}'.format(comp[i])
        coff = results.loc[results.Variable == new[i] , 'Coefficient'].tolist()[0]
        temp = df[com] * coff / df['std']
        new_f = pd.DataFrame({'Comp':[new[i]]*len(df),'Var':group.iloc[var_id-1,:].dropna().values.tolist(),'Coef':temp.tolist()})
        if 'const' in results['Variable'].tolist() or 'Intercept' in results['Variable'].tolist():
            try:
                con = results.loc[results.Variable == 'const', 'Coefficient'].tolist()[0]
            except:
                con = results.loc[results.Variable == 'Intercept', 'Coefficient'].tolist()[0]
            intercept = sum(temp * df['Mean']) + con
            inter = pd.DataFrame({'Intercept': [intercept]})
            new_f = pd.concat([new_f, inter], axis=1)
            unwind = pd.concat([unwind, new_f], ignore_index=True)
        else:
            unwind = pd.concat([unwind, new_f], ignore_index=True)
    return unwind.to_html(classes= 'equ')


#### Model Validation Lift Curve ####
def drawliftcure(libname, predict_df, y_name, i):
    deciles = np.arange(0,110,10)
    b = predict_df.sort_values(by = 'b')['b']
    a = predict_df.columns[0]
    percentile = [np.percentile(b.values, m) for m in deciles]
    lift_index = []
    for m in range(0,10):
        lift_index.append(b[np.all([b.values <= percentile[m+1], b.values >= percentile[m]], axis =0)].index)
    # calculate the lift
    actual = [np.mean(predict_df.loc[lift_index[i],a]) for i in range(0,10)]
    # actual_ = np.asarray(actual)/len(regout[a])
    predict = [np.mean(predict_df.loc[lift_index[i],'b']) for i in range(0,10)]
    # predict_ = np.asarray(predict)/len(regout[b])
    # plot the lift chart
    x_axis = np.arange(1,11,1)
    plt.plot(x_axis, actual,'cs-',label = 'Actual')
    plt.plot(x_axis,predict,'r^-',label = 'Predict')
    plt.legend(loc='upper left')
    plt.title('Lift Curve')
    plt.xlabel('Decile')
    plt.xticks([1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0])
    fig = plt.gcf()
    fig.set_size_inches(10,8)
    figname = '{}/lift_{}{}.png'.format(libname,i + 1, y_name)
    plt.savefig(figname)
    plt.close('all')
    return


#### Roc Auc #####
def roc(libname, x, y_mn, y_n, results, level, i, y_name, modeltype):
    y_score= results.predict(x).values
    lw = 2
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    try:
        for a in range(level):
            fpr[a], tpr[a], _ = roc_curve(y_mn[:, a], y_score[:, a])
            roc_auc[a] = auc(fpr[a], tpr[a])
        fpr["micro"], tpr["micro"], _ = roc_curve(y_mn.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(level)]))
        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for a in range(level):
            mean_tpr += interp(all_fpr, fpr[a], tpr[a])
        # Finally average it and compute AUC
        mean_tpr /= level
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
        # Plot all ROC curves
        plt.figure()
        plt.plot(fpr["micro"], tpr["micro"],
                 label='micro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["micro"]),
                 color='deeppink', linestyle=':', linewidth=4)
        plt.plot(fpr["macro"], tpr["macro"],
                 label='macro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["macro"]),
                 color='navy', linestyle=':', linewidth=4)
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
        for a, color in zip(range(level), colors):
            plt.plot(fpr[a], tpr[a], color=color, lw=lw,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                     ''.format(a, roc_auc[a]))
        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    except:
        fpr, tpr, thresholds = metrics.roc_curve(y_n, y_score)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Model%d %s' %(i + 1, y_name))
    plt.legend(loc="lower right")
    plotname = '{}/roc_{}{}.png'.format(libname,i + 1, y_name)
    plt.savefig(plotname)
    plt.close('all')
    return


############### GLM & Report #################
def do_genmod(libname, CappedData_in, y_in, regression_out, PCAGroup, Summary,x_in= None, family=None,link=None, alpha = 1.0, var_power = 1.0, power =1.0, offset=None, var_weights = None, NAstrategy='mean', method = 'group', interact=None, drop = None, modeltype= 'Prediction', maxiter=1000, plot = None , split = 0.8, sampling='False'):
    logger = mylog('do_genmod')
    # load input variables involved in the glm analysis
    try:
        x_name_in = "{}/{}.csv".format(libname, x_in)
        predictor = pd.read_csv(x_name_in, encoding='utf-8', header=None)
    except:
        predictor = []
    # load response variables involved in the glm analysis
    try:
        y_name_in = "{}/{}.csv".format(libname, y_in)
        response_y = pd.read_csv(y_name_in, encoding='utf-8', header=None)
    except:
        response_y = [y_in]
    df_name = "{}/{}.csv".format(libname, CappedData_in)
    df = pd.read_csv(df_name, encoding='utf-8')
    # load interaction variables
    if interact != None:
        interact_name = "{}/{}.csv".format(libname, interact)
        interact_in = pd.read_csv(interact_name, encoding='utf-8',header=None)
        interact = [interact_in.iloc[i, :].dropna() for i in range(len(interact_in))]
        inter = ['*'.join(['C({})'.format(j) for j in x]) for x in interact]
    else:
        inter =[]
        logger.info('No Interaction Variable Input & No Variable Dropped')
    # load drop variables
    if drop != None:
        try:
            drop_name = "{}/{}.csv".format(libname, drop)
            drop_var = pd.read_csv(drop_name, encoding='utf-8',header=None).values.tolist()
        except:
            drop_var = []
    else:
        drop_var =[]
        logger.info('No Variable Dropped')
    group_name = "{}/{}.csv".format(libname, PCAGroup)
    group = pd.read_csv(group_name, encoding='utf-8', header=None)
    sum_name = "{}/{}.csv".format(libname, Summary)
    summ = pd.read_csv(sum_name, encoding='utf-8', index_col=0)
    summ = summ.reset_index(drop=True)
    out_name = "{}/{}.csv".format(libname, regression_out)
    # determine the parent class for one-parameter exponential families
    if link is 'log':
        link = sm.families.links.log
    elif link is 'logit':
        link = sm.families.links.logit
    elif link is 'power':
        link = sm.families.links.power(power= power)
    elif link is 'identity':
        link = sm.families.links.identity
    elif link is 'inverse':
        link = sm.families.links.inverse_power
    elif link is 'sqrt':
        link = sm.families.links.power(power= 0.5)
    elif link is 'probit':
        link = sm.families.links.probit
    elif link is 'cauchy':
        link = sm.families.links.cauchy
    elif link is 'cloglog':
        link = sm.families.links.cloglog
    elif link is 'inverse_squared':
        link = sm.families.links.power(power= -2.)
    elif link is 'nbinom':
        link = sm.families.links.nbinom(alpha=alpha)
    if family is 'Gaussian':
        family = sm.families.Gaussian(link=link)
    elif family is 'Inverse Gaussian':
        family = sm.families.InverseGaussian(link=link)
    elif family is 'Binomial':
        family = sm.families.Binomial(link=link)
    elif family is 'Negative Binomial':
        family = sm.families.NegativeBinomial(link=link, alpha=alpha)
    elif family is 'Possion':
        family = sm.families.Possion(link=link)
    elif family is 'Gamma':
        family = sm.families.Gamma(link=link)
    elif family is 'Tweedie':
        family = sm.families.Tweedie(link=link, var_power=var_power)
    else:
        family = sm.families.Gaussian()
    out1 = []
    out2 = pd.DataFrame()
    param = pd.DataFrame()
    unique = pd.DataFrame()
    if method is 'all':
        logger.info('Modeling with all predictor variables')
        try:
            response = list(chain(*response_y.values))
        except:
            response = response_y
        remaining = list(set(df.columns.tolist()).difference(set(response)))
        remaining = [i for i in remaining if i not in df.select_dtypes(include=['object']).columns.tolist()]
        for i in response:
            loop = 0
            logger.info('Response Variable %s is used' %i)
            try:
                drop = drop_var[loop]
            except:
                drop = []
            remaining = list(set(remaining).difference(set(drop)))
            temp_remaining = remaining + [i]
            temp_df = df.dropna(subset= [i])
            temp_df = temp_df.filter(items= temp_remaining)
            imr = Imputer(missing_values='NaN', strategy=NAstrategy, axis=0)
            logger.info('"%s" strategy is used for imputing missing data' %(NAstrategy))
            imputed_df = pd.DataFrame(imr.fit_transform(temp_df))
            imputed_df.columns = temp_df.columns
            imputed_df.index = temp_df.index
            if modeltype is 'Prediction':
                # divide the training and testing sets
                imputed_train, imputed_test = train_test_split(imputed_df, train_size= split, random_state=0)
                try:
                    formula = '{} ~ {} + {}'.format(str(i), ' + '.join(remaining), ' + '.join(inter))
                    results = smf.glm(formula, imputed_train ,family=family).fit(maxiter= maxiter)
                    logger.info('Fomula: %s ' %formula)
                except:
                    formula = '{} ~ {}'.format(str(i), ' + '.join(remaining))
                    results = smf.glm(formula, imputed_train ,family=family).fit(maxiter= maxiter)
                    logger.info('Fomula: %s ' %formula)
                    logger.info('No Interaction Variable Input')
                results.save("{}/{}_{}.pickle".format(libname, modeltype, i))
                a = results.summary2().as_html()
                result_param = results.params.reset_index()
                result_param = result_param.rename(columns={'index': 'Variable', 0: 'Coefficient'})
                # restore coefficients to orignal variables
                if True in [('New_Var' in q) for q in temp_remaining]:
                    out = unwindparam(remaining, group, summ, result_param, logger)
                    logger.info('PCA variables are being unwinded and added to the model')
                else:
                    out = []
                    logger.info('No PCA variables are being unwinded')
                colname = 'all.{}'.format(str(i))
                tmp = result_param.rename(columns={'Variable': colname + '_variable', 'Coefficient': colname + '_coefficient'})
                param = pd.concat([param, tmp], axis=1)
            elif modeltype is 'MNClassification' and len(imputed_df[i].unique()) >2:
                level = len(imputed_df[i].unique())
                temp_res, temp_unique = pd.factorize(imputed_df[i])
                tmp_list = imputed_df[i].values.tolist()
                label_seq = list(set(tmp_list))
                label_seq.sort(key=tmp_list.index)
                imputed_df.loc[:,i] = temp_res
                # divide the training and testing sets
                imputed_train, imputed_test = train_test_split(imputed_df, train_size=split, random_state=0)
                # adopt sampling
                if sampling == 'True':
                    over_samples = SMOTE(ratio='auto', random_state=0)
                    over_samples_X, over_samples_Y = over_samples.fit_sample(imputed_train[remaining], imputed_train[i])
                    imputed_train = pd.DataFrame(over_samples_X, columns=remaining)
                    imputed_train[i] = pd.Series(over_samples_Y)
                formula = "{} ~ {} ".format(str(i), ' + '.join(remaining))
                logger.info('Fomula: %s ' %formula)
                try:
                    results = smf.mnlogit(formula, imputed_train).fit(maxiter=maxiter)
                    unique = pd.concat([unique, pd.DataFrame({'Before_{}'.format(str(i)) : label_seq,'After_{}'.format(str(i)): list(range(level))})], axis = 1)
                except:
                    logger.info('Dummy Variable trap encountered,drop intercept')
                    formula = "{} ~ {}-1 ".format(str(i), ' + '.join(remaining))
                    logger.info('Fomula: %s ' %formula)
                    try:
                        results = smf.mnlogit(formula, imputed_train).fit(maxiter=maxiter)
                        unique = pd.concat([unique, pd.DataFrame({'Before_{}'.format(str(i)) : label_seq,'After_{}'.format(str(i)): list(range(level))})], axis = 1)
                    except:
                        results = smf.glm(formula, imputed_train, family=sm.families.Binomial()).fit(maxiter=maxiter)
                        logger.exception(sys.exc_info())
                        logger.info('Response Variable does not fit Logistic Regression model, use glm instead')
                results.save("{}/{}_{}.pickle".format(libname, modeltype, i))
                a = results.summary().as_html()
                result_param = results.params.reset_index()
                colname = 'all.{}'.format(str(i))
                result_param = result_param.rename(columns={'index': colname + '_variable'})
                for k in range(len(temp_unique) - 1):
                    result_param = result_param.rename(columns={k: colname + '_coefficient' + str(temp_unique.tolist()[k + 1]) + '/' + str(temp_unique.tolist()[0])})
                out = []
                param = pd.concat([param, result_param], axis=1)
            # level = 2
            elif modeltype is 'Classification' and len(imputed_df[i].unique()) <=2:
                level = len(imputed_df[i].unique())
                temp_res, temp_unique = pd.factorize(imputed_df[i])
                imputed_df.loc[:,i] = temp_res
                tmp_list = imputed_df[i].values.tolist()
                label_seq = list(set(tmp_list))
                label_seq.sort(key=tmp_list.index)
                # divide the training and testing sets
                imputed_train, imputed_test = train_test_split(imputed_df, train_size=split, random_state=0)
                # adopt sampling
                if sampling == 'True':
                    over_samples = SMOTE(ratio='auto', random_state=0)
                    over_samples_X, over_samples_Y= over_samples.fit_sample(imputed_train[remaining], imputed_train[i])
                    imputed_train = pd.DataFrame(over_samples_X, columns=remaining)
                    imputed_train[i] = pd.Series(over_samples_Y)
                formula = "{} ~ {} ".format(str(i), ' + '.join(remaining))
                logger.info('Fomula: %s ' %formula)
                try:
                    results = smf.logit(formula, imputed_train).fit(maxiter=maxiter)
                    unique = pd.concat([unique, pd.DataFrame({'Before_{}'.format(str(i)) : label_seq, 'After_{}'.format(str(i)): list(range(level))})], axis = 1)
                except:
                    logger.info('Dummy Variable trap encountered,drop intercept')
                    formula = "{} ~ {}-1 ".format(str(i), ' + '.join(remaining))
                    logger.info('Fomula: %s ' %formula)
                    try:
                        results = smf.logit(formula, imputed_train).fit(maxiter=maxiter)
                        unique = pd.concat([unique, pd.DataFrame({'Before_{}'.format(str(i)) : label_seq,'After_{}'.format(str(i)): list(range(level))})], axis = 1)
                    except:
                        results = smf.glm(formula, imputed_train, family=sm.families.Binomial()).fit(maxiter=maxiter)
                        logger.exception(sys.exc_info())
                        logger.info('Response Variable does not fit Logistic Regression model, use glm instead')
                results.save("{}/{}_{}.pickle".format(libname, modeltype, i))
                a = results.summary2().as_html()
                result_param = results.params.reset_index()
                result_param = result_param.rename(columns={'index': 'Variable', 0: 'Coefficient'})
                # restore coefficients to orignal variables
                if True in [('New_Var' in q) for q in remaining]:
                    out = unwindparam(remaining, group, summ, result_param, logger)
                    logger.info('PCA variables are being unwinded and added to the model')
                else:
                    out = []
                    logger.info('No PCA variables are being unwinded')
                colname = 'all.{}'.format(str(i))
                tmp = result_param.rename(columns={'Variable': colname + '_variable', 'Coefficient': colname + '_coefficient'})
                param = pd.concat([param, tmp], axis=1)
            # predict results
            x_test = imputed_test.drop([i], axis=1)
            y_test = imputed_test[i]
            inx = imputed_test.index.tolist()
            if modeltype is 'Prediction':
                predicted = results.predict(x_test)
            elif modeltype is 'MNClassification' :
                if isinstance(results.model, statsmodels.discrete.discrete_model.MNLogit) == True:
                    y_score = results.predict(x_test)
                    y_score_class = y_score.idxmax(axis=1).values
                else:
                    predicted = results.predict(x_test)
            elif modeltype is 'Classification':
                if isinstance(results.model, statsmodels.discrete.discrete_model.Logit) == True:
                    predicted_score = results.predict(x_test)
                    predicted_class = np.array([1 if x > 0.5 else 0 for x in predicted_score])
                else:
                    predicted = results.predict(x_test)
            # model valuation
            if modeltype is 'Prediction':
                y_estimate = pd.DataFrame({'Model': '{}_{}'.format(modeltype,i), 'explained_variance': metrics.explained_variance_score(y_test, predicted),
                                           'mean_absolute_error': metrics.mean_absolute_error(y_test, predicted), 'mean_squared_error': metrics.mean_squared_error(y_test, predicted),
                                           'root_mean_squared_error': np.sqrt(metrics.mean_squared_error(y_test, predicted)), 'r2': metrics.r2_score(y_test, predicted)}, index=[0])
            elif modeltype is 'MNClassification':
                if isinstance(results.model, statsmodels.discrete.discrete_model.MNLogit) == True:
                    y_test_b = label_binarize(y_test.values, classes=list(range(level)))
                    y_test_p = label_binarize(y_score_class, classes=list(range(level)))
                    y_estimate = pd.DataFrame({'Model': '{}_{}'.format(modeltype, i), 'average_precision_score': metrics.average_precision_score(y_test_b,y_score),
                                               'coverage_error': metrics.coverage_error(y_test_b,y_score),'label_ranking_average_precision_score': metrics.label_ranking_average_precision_score(y_test_b,y_score),
                                               'label_ranking_loss': metrics.label_ranking_loss(y_test_b,y_score),'accuracy_score': metrics.accuracy_score(y_test_b, y_test_p),
                                               'f1_score': metrics.f1_score(y_test, y_score_class, average='macro'), 'precision_score': metrics.precision_score(y_test, y_score_class, average='macro'),
                                               'recall_score': metrics.recall_score(y_test, y_score_class, average='macro'),'roc_auc_score':metrics.roc_auc_score(y_test_b, y_score)}, index=[0])
                else:
                    y_estimate = pd.DataFrame({'Model': '{}_{}'.format(modeltype, i),'explained_variance': metrics.explained_variance_score(y_test,predicted),
                                               'mean_absolute_error': metrics.mean_absolute_error(y_test, predicted), 'mean_squared_error': metrics.mean_squared_error(y_test, predicted),
                                               'root_mean_squared_error': np.sqrt(metrics.mean_squared_error(y_test, predicted)), 'r2': metrics.r2_score(y_test, predicted)}, index=[0])
            elif modeltype is 'Classification':
                if isinstance(results.model, statsmodels.discrete.discrete_model.Logit) == True:
                    y_estimate = pd.DataFrame({'Model': '{}_{}'.format(modeltype, i), 'accuracy_score': metrics.accuracy_score(y_test,predicted_class),
                                               'average_precision_score': metrics.average_precision_score(y_test,predicted_score), 'f1_score': metrics.f1_score(y_test,predicted_class),
                                               'precision_score': metrics.precision_score(y_test,predicted_class), 'recall_score': metrics.recall_score(y_test,predicted_class),
                                               'roc_auc_score':metrics.roc_auc_score(y_test,predicted_score)}, index=[0])
                else:
                    y_estimate = pd.DataFrame({'Model': '{}_{}'.format(modeltype, i),'explained_variance': metrics.explained_variance_score(y_test,predicted),
                                               'mean_absolute_error': metrics.mean_absolute_error(y_test, predicted), 'mean_squared_error': metrics.mean_squared_error(y_test, predicted),
                                               'root_mean_squared_error': np.sqrt(metrics.mean_squared_error(y_test, predicted)), 'r2': metrics.r2_score(y_test, predicted)}, index=[0])
            out1.append([[i], a, out])
            # match the true and predicted values of y
            out2 = pd.concat([out2, y_estimate], ignore_index=True, sort=False)
            # Plot
            if plot is 'Roc' and modeltype != 'Prediction':
                try:
                    y_mn = label_binarize(y_test, classes=list(range(level)))
                    y_n = y_test
                    roc(libname = libname, x = x_test, y_mn=y_mn, y_n=y_n , results=results,level=level,i= loop,y_name=str(i),modeltype=modeltype)
                    logger.info('Roc-Auc Curve for Response Variable %s is generated' %i)
                except:
                    logger.exception(sys.exc_info())
                    logger.info('Roc-Auc Curve for Response Variable %s cannot be generated' %i)
            elif plot is 'Lift' and modeltype is 'Prediction':
                predict_df = pd.concat([y_test, pd.DataFrame(predicted, index= inx, columns =['b'])],axis=1)
                drawliftcure(libname=libname,predict_df=predict_df,y_name = str(i),i=loop)
                logger.info('Lift Curve for Response Variable "%s" is generated' %i)
            loop += 1
    elif method is 'group':
        logger.info('Modeling with Defined group predictors')
        for i in range(0, len(predictor)):
            pre = predictor.iloc[i].dropna().tolist()
            for j in range(0, len(response_y)):
                res = response_y.iloc[j].tolist()
                y_name = ''.join(res)
                # drop rows without y
                temp_df = df.dropna(subset= res)
                y_del = list(set(response_y[0].tolist()).difference(set(res)))
                # filter string variables
                x_remain = temp_df.columns.tolist()
                x_remain = [item for item in x_remain if item not in y_del and item not in temp_df.select_dtypes(include=['object']).columns.tolist()]
                temp_df = temp_df.filter(items= x_remain)
                # fill in missing values
                imr = Imputer(missing_values='NaN', strategy = NAstrategy, axis=0)
                logger.info('"%s" strategy is used for imputing missing data' % (NAstrategy))
                imputed_df = pd.DataFrame(imr.fit_transform(temp_df))
                imputed_df.columns = temp_df.columns
                imputed_df.index = temp_df.index
                logger.info('Model between Group %d-th predictor variables and %d-th response variable is being fitted'%(i,j))
                if modeltype is 'Prediction' :
                    # divide the training and testing sets
                    imputed_train, imputed_test = train_test_split(imputed_df, train_size=0.8, random_state=0)
                    try:
                        pre = [item for item in pre if item not in drop_var]
                        formula = '{} ~ {} + {}'.format(y_name, ' + '.join(pre), ' + '.join(inter))
                        logger.info('Fomula: %s ' %formula)
                        results = smf.glm(formula, imputed_train ,family=family).fit(maxiter= maxiter)
                    except:
                        formula = '{} ~ {} '.format(y_name, ' + '.join(pre))
                        logger.info('Fomula: %s ' %formula)
                        results = smf.glm(formula, imputed_train ,family=family).fit(maxiter= maxiter)
                    results.save("{}/{}_{}{}.pickle".format(libname, modeltype, y_name, i+1))
                    a = results.summary2().as_html()
                    result_param = results.params.reset_index()
                    result_param = result_param.rename(columns={'index': 'Variable', 0: 'Coefficient'})
                    # restore coefficients to orignal variables
                    if True in [('New_Var' in q) for q in pre]:
                        out = unwindparam(pre, group, summ, result_param, logger)
                        logger.info('PCA variables are being unwinded and added to the model')
                    else:
                        out = []
                        logger.info('No PCA variables are being unwinded')
                    colname = 'x_in.{}.{}'.format(str(i + 1), y_name)
                    tmp = result_param.rename(columns={'Variable': colname + '_variable', 'Coefficient': colname + '_coefficient'})
                    param = pd.concat([param, tmp], axis=1)
                # level > 2
                elif modeltype is 'MNClassification' and len(imputed_df[y_name].unique()) >2:
                    level = len(imputed_df[y_name].unique())
                    temp_res, temp_unique = pd.factorize(imputed_df[y_name])
                    tmp_list = imputed_df[y_name].values.tolist()
                    label_seq = list(set(tmp_list))
                    label_seq.sort(key=tmp_list.index)
                    imputed_df.loc[:,y_name] = temp_res
                    # divide the training and testing sets
                    imputed_train, imputed_test = train_test_split(imputed_df, train_size=0.8, random_state=0)
                    # adopt sampling
                    if sampling == 'True':
                        over_samples = SMOTE(ratio='auto', random_state=0)
                        over_samples_X, over_samples_Y = over_samples.fit_sample(imputed_train[pre],imputed_train[y_name])
                        imputed_train = pd.DataFrame(over_samples_X, columns=pre)
                        imputed_train[y_name] = pd.Series(over_samples_Y)
                    formula = "{} ~ {} ".format(y_name,' + '.join(pre))
                    logger.info('Fomula: %s ' %formula)
                    try:
                        results = smf.mnlogit(formula, imputed_train).fit(maxiter= maxiter)
                        unique = pd.concat([unique, pd.DataFrame({'Before_{}{}'.format(str(y_name),i+1) : label_seq,'After_{}{}'.format(str(y_name), i+1): list(range(level))})], axis = 1)
                    except:
                        logger.info('Dummy Variable trap encountered,drop intercept')
                        formula = "{} ~ {}-1 ".format(y_name,' + '.join(pre))
                        logger.info('Fomula: %s ' %formula)
                        try:
                            results = smf.mnlogit(formula, imputed_train).fit(maxiter=maxiter)
                            unique = pd.concat([unique, pd.DataFrame({'Before_{}{}'.format(str(y_name),i+1) : label_seq,'After_{}{}'.format(str(y_name), i+1): list(range(level))})], axis = 1)
                        except:
                            results = smf.glm(formula, imputed_train, family=sm.families.Binomial()).fit(maxiter=maxiter)
                            logger.exception(sys.exc_info())
                            logger.info('Response Variable does not fit Logistic Regression model, use glm instead')
                    results.save("{}/{}_{}{}.pickle".format(libname, modeltype, y_name, i+1))
                    a = results.summary().as_html()
                    result_param = results.params.reset_index()
                    colname = 'x_in.{}.{}'.format(str(i + 1), y_name)
                    result_param = result_param.rename(columns={'index': colname + '_variable'})
                    for k in range(len(temp_unique)-1):
                        result_param = result_param.rename(columns={k : colname + '_coefficient' + str(temp_unique.tolist()[k+1]) + '/' + str(temp_unique.tolist()[0])})
                    out = []
                    param = pd.concat([param, result_param], axis=1)
                # level = 2
                elif modeltype is 'Classification' and len(imputed_df[y_name].unique()) <=2:
                    level = len(imputed_df[y_name].unique())
                    temp_res, temp_unique = pd.factorize(imputed_df[y_name])
                    tmp_list = imputed_df[y_name].values.tolist()
                    label_seq = list(set(tmp_list))
                    label_seq.sort(key=tmp_list.index)
                    imputed_df.loc[:,y_name] = temp_res
                    # divide the training and testing sets
                    imputed_train, imputed_test = train_test_split(imputed_df, train_size=0.8, random_state=0)
                    # adopt sampling
                    if sampling == 'True':
                        over_samples = SMOTE(ratio='auto', random_state=0)
                        over_samples_X, over_samples_Y = over_samples.fit_sample(imputed_train[pre],imputed_train[y_name])
                        imputed_train = pd.DataFrame(over_samples_X, columns=pre)
                        imputed_train[y_name] = pd.Series(over_samples_Y)
                    formula = "{} ~ {} ".format(y_name,' + '.join(pre))
                    logger.info('Fomula: %s ' %formula)
                    try:
                        results = smf.logit(formula, imputed_train).fit(maxiter=maxiter)
                        unique = pd.concat([unique, pd.DataFrame({'Before_{}{}'.format(str(y_name),i+1) : label_seq,'After_{}{}'.format(str(y_name), i+1): list(range(level))})], axis = 1)
                    except:
                        logger.info('Dummy Variable trap encountered,drop intercept')
                        formula = "{} ~ {}-1 ".format(y_name,' + '.join(pre))
                        logger.info('Fomula: %s ' %formula)
                        try:
                            results = smf.logit(formula, imputed_train).fit(maxiter=maxiter)
                            unique = pd.concat([unique, pd.DataFrame({'Before_{}{}'.format(str(y_name),i+1) : label_seq,'After_{}{}'.format(str(y_name), i+1): list(range(level))})], axis = 1)
                        except:
                            results = smf.glm(formula, imputed_train, family=sm.families.Binomial()).fit(maxiter=maxiter)
                            logger.exception(sys.exc_info())
                            logger.info('Response Variable does not fit Logistic Regression model, use glm instead')
                    results.save("{}/{}_{}{}.pickle".format(libname, modeltype, y_name, i+1))
                    a = results.summary2().as_html()
                    result_param = results.params.reset_index()
                    result_param = result_param.rename(columns={'index': 'Variable', 0: 'Coefficient'})
                    # restore coefficients to orignal variables
                    if True in [('New_Var' in q) for q in pre]:
                        out = unwindparam(pre, group, summ, result_param, logger)
                        logger.info('PCA variables are being unwinded and added to the model')
                    else:
                        out = []
                        logger.info('No PCA variables are being unwinded')
                    colname = 'x_in.{}.{}'.format(str(i + 1), y_name)
                    tmp = result_param.rename(columns={'Variable': colname + '_variable', 'Coefficient': colname + '_coefficient'})
                    param = pd.concat([param, tmp], axis=1)
                # predict results
                x_test = imputed_test.drop([y_name], axis=1)
                y_test = imputed_test[y_name]
                inx = imputed_test.index.tolist()
                if modeltype is 'Prediction':
                    predicted = results.predict(x_test)
                elif modeltype is 'MNClassification' :
                    if isinstance(results.model, statsmodels.discrete.discrete_model.MNLogit) == True:
                        y_score = results.predict(x_test)
                        y_score_class = y_score.idxmax(axis=1).values
                    else:
                        predicted = results.predict(x_test)
                elif modeltype is 'Classification':
                    if isinstance(results.model, statsmodels.discrete.discrete_model.Logit) == True:
                        predicted_score = results.predict(x_test)
                        predicted_class = np.array([1 if x > 0.5 else 0 for x in predicted_score])
                    else:
                        predicted = results.predict(x_test)
                # model valuation
                if modeltype is 'Prediction':
                    y_estimate = pd.DataFrame({'Model': '{}_{}.x_in.{}'.format(modeltype, y_name, str(i + 1)),'explained_variance': metrics.explained_variance_score(y_test,predicted),
                                               'mean_absolute_error': metrics.mean_absolute_error(y_test, predicted),'mean_squared_error': metrics.mean_squared_error(y_test, predicted),
                                               'root_mean_squared_error': np.sqrt(metrics.mean_squared_error(y_test, predicted)),'r2': metrics.r2_score(y_test, predicted)}, index=[0])
                elif modeltype is 'MNClassification':
                    if isinstance(results.model, statsmodels.discrete.discrete_model.MNLogit) == True:
                        y_test_b = label_binarize(y_test.values, classes=list(range(level)))
                        y_test_p = label_binarize(y_score_class, classes=list(range(level)))
                        y_estimate = pd.DataFrame({'Model': '{}_{}.x_in.{}'.format(modeltype, y_name, str(i + 1)), 'average_precision_score': metrics.average_precision_score(y_test_b,y_score),
                                                   'coverage_error': metrics.coverage_error(y_test_b, y_score),'label_ranking_average_precision_score': metrics.label_ranking_average_precision_score(y_test_b, y_score),
                                                   'label_ranking_loss': metrics.label_ranking_loss(y_test_b, y_score), 'accuracy_score': metrics.accuracy_score(y_test_b, y_test_p),
                                                   'f1_score': metrics.f1_score(y_test, y_score_class, average='macro'),'precision_score': metrics.precision_score(y_test, y_score_class,average='macro'),
                                                   'recall_score': metrics.recall_score(y_test, y_score_class,average='macro'),'roc_auc_score': metrics.roc_auc_score(y_test_b, y_score)}, index=[0])
                    else:
                        y_estimate = pd.DataFrame({'Model': '{}_{}.x_in.{}'.format(modeltype, y_name, str(i + 1)),'explained_variance': metrics.explained_variance_score(y_test,predicted),
                                                   'mean_absolute_error': metrics.mean_absolute_error(y_test, predicted), 'mean_squared_error': metrics.mean_squared_error(y_test, predicted),
                                                   'root_mean_squared_error': np.sqrt(metrics.mean_squared_error(y_test, predicted)), 'r2': metrics.r2_score(y_test, predicted)}, index=[0])
                elif modeltype is 'Classification':
                    if isinstance(results.model, statsmodels.discrete.discrete_model.Logit) == True:
                        y_estimate = pd.DataFrame({'Model': '{}_{}.x_in.{}'.format(modeltype, y_name, str(i + 1)), 'accuracy_score': metrics.accuracy_score(y_test, predicted_class),
                                                   'average_precision_score': metrics.average_precision_score(y_test,predicted_score),
                                                   'f1_score': metrics.f1_score(y_test, predicted_class),'precision_score': metrics.precision_score(y_test, predicted_class),
                                                   'recall_score': metrics.recall_score(y_test, predicted_class),'roc_auc_score': metrics.roc_auc_score(y_test, predicted_score)},index=[0])
                    else:
                        y_estimate = pd.DataFrame({'Model': '{}_{}.x_in.{}'.format(modeltype, y_name, str(i + 1)), 'explained_variance': metrics.explained_variance_score(y_test,predicted),
                                                   'mean_absolute_error': metrics.mean_absolute_error(y_test,predicted),'mean_squared_error': metrics.mean_squared_error(y_test, predicted),
                                                   'root_mean_squared_error': np.sqrt(metrics.mean_squared_error(y_test, predicted)),'r2': metrics.r2_score(y_test, predicted)}, index=[0])
                out1.append([res, a, out])
                # match the true and predicted values of y
                out2 = pd.concat([out2, y_estimate], ignore_index=True, sort=False)
                # Plot
                if plot is 'Roc' and modeltype != 'Prediction':
                    try:
                        y_mn = label_binarize(y_test, classes=list(range(level)))
                        y_n = y_test
                        roc(libname = libname, x = x_test, y_mn=y_mn,y_n=y_n , results=results,level=level,i=i,y_name=y_name,modeltype=modeltype)
                        logger.info('Roc-Auc Curve for Response Variable %s is generated' %j)
                    except:
                        logger.exception(sys.exc_info())
                        logger.info('Roc-Auc Curve for Response Variable %s cannot be generated' %i)
                elif plot is 'Lift' and modeltype is 'Prediction':
                    predict_df = pd.concat([y_test,pd.DataFrame(predicted, index= inx, columns =['b'])],axis=1)
                    drawliftcure(libname=libname, predict_df=predict_df, y_name = y_name, i=i)
                    logger.info('Lift Curve for Response Variable %s is generated' %j)
    out2.to_csv(out_name, index=False)
    logger.info('The Orignal/Prediction file is being saved')
    param.to_csv('{}/param.csv'.format(libname), index=False)
    logger.info('The Parameter file is being saved')
    unique.to_csv('{}/Factorize.csv'.format(libname), index=False)
    return out1


def app3(libname, CappedData_in, y_in, regression_out, PCAGroup, Summary, x_in= None,family=None,link=None, alpha = 1.0, var_power = 1.0, power =1.0,offset=None, var_weights = None, NAstrategy='mean', method = 'group', interact= None, drop=None, modeltype= 'Prediction', maxiter=1000, plot = None,split=0.8, sampling='False'):
    app = Flask(__name__)

    @app.route('/')
    def index():
        out3 = [do_genmod(libname, CappedData_in, y_in, regression_out, PCAGroup, Summary,x_in,family,link, alpha, var_power, power,offset, var_weights, NAstrategy, method , interact, drop, modeltype, maxiter, plot,split, sampling)]
        return render_template('template3.html', Var=out3[0], title='GLM Report')

    @app.template_global('current_time')
    def current_time(timeFormat="%b %d, %Y - %H:%M:%S"):
        return time.strftime(timeFormat)

    if __name__ == 'myfun':
        app.run(debug=True)
    return


############### Prediction #################
def loadmodel(libname, modelname,modeltype, new_data_tofit, factorize= None):
    new_df = pd.read_csv("{}/{}.csv".format(libname, new_data_tofit), encoding='utf-8')
    results = load_pickle("{}/{}.pickle".format(libname, modelname))
    predicted = results.predict(new_df)
    if modeltype is 'Prediction':
        predicted = predicted
    elif modeltype is 'MNClassification':
        pre_df = pd.DataFrame(data= predicted, columns = factorize)
        pre_tu= pre_df.idxmax(axis=1).values
        predicted = np.array(pre_tu)
    elif modeltype is 'Classification':
        factorize.sort()
        predicted = np.array([int(factorize[1]) if x > 0.5 else int(factorize[0]) for x in predicted])
    out = pd.DataFrame({'y': predicted})
    out.to_csv('{}/Predict_Y.csv'.format(libname), index=False)
    return
