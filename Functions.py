import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score, accuracy_score
from sklearn.utils import resample
from scipy.stats import shapiro, ttest_ind, mannwhitneyu#, anderson_ksamp
from scipy.stats import zscore
from statsmodels.stats.multitest import multipletests
import random
from tqdm import tqdm 



def cargarLimpiarDataFrame(url = '../mRNA_seq/mRNA_seq_AS_Control_LncRNA_identificados.xlsx'):
    df_raw = pd.read_excel(url)
    #Eliminamos todas las variables que Norm ya que prefiero hacer nuestra propia normalizacion
    df_raw = df_raw.loc[:, ~df_raw.columns.str.contains('Norm')]
    #trasponemos para que cada fila sea un paciente y cada columna sea una variable , incluimos el simbolo de hgnc como nombre de columnas  y eliminamos variables que son string
    df_raw = df_raw.set_index('hgnc_symbol').drop(['description','gene_biotype','chromosome_name'],axis = 1).T
    df_raw_normalizado = (df_raw - df_raw.mean())/df_raw.std()
    return df_raw_normalizado

def eliminarLecturasBajas(df, lecturaBaja = 0.6, proporcionLecturasBajas = 0.5):
    estadisticas = pd.DataFrame({
    'NoCeros': (df != 0).sum(),
    'CercanoCero': (df < lecturaBaja).sum(),
    'Porcentaje de lencturas cercanas a cero' : (df < lecturaBaja).sum()/35,
    'Media': df.mean(),
    'Desviación': df.std(),
    'Suma total': df.sum(),
    'Maximo' : df.max(),
    'Minimo' : df.min()
    }).sort_values('Porcentaje de lencturas cercanas a cero', ascending = False)
    vars_est = estadisticas[estadisticas['Porcentaje de lencturas cercanas a cero'] > proporcionLecturasBajas]
    vars_excluir = vars_est.reset_index()['hgnc_symbol'].to_list()
    df1 = df.drop(columns=vars_excluir)

    print(len(df.columns),'variables →',len(df1.columns),'variables')
    return df

def eliminarCorrelacionesAltas(df,threshold = 0.8):
    correlaciones_vars = df.corr(method = 'spearman') #usar mejor Spearman, rank correlation
    columns_to_drop = set()

    # Iterar sobre la matriz de correlación y eliminar las columnas con correlación > 0.95
    for i in range(len(correlaciones_vars.columns)):
        for j in range(i):
            if abs(correlaciones_vars.iloc[i, j]) > threshold:
                colname = correlaciones_vars.columns[i]
                columns_to_drop.add(colname)
    nvarsAntes = len(df.columns)        
    df = df.drop(columns=columns_to_drop)   
    print(nvarsAntes,'variables →',len(df.columns),'variables')
    return df

def eliminarTests (df):
    ####Dividmos el data set en dos: Uno de tratamiento y otro de control

    df_ct = df.copy().reset_index()
    df_ct['Grupo'] = np.where(df_ct['index'].str.contains('A'), 'Tratamiento', 
                np.where(df_ct['index'].str.contains('C'), 'Control', 'Otro'))
    df_ct = df_ct.set_index('index')
    df_trnto = df_ct[df_ct['Grupo'] == 'Tratamiento'].drop('Grupo', axis = 1)
    df_ctrl = df_ct[df_ct['Grupo'] == 'Control'].drop('Grupo', axis = 1)
    df_ct = df_ct.drop('Grupo', axis = 1)
    variables = df_ct.columns  
    p_values = []
    tests = []  # Para almacenar qué test se usó en cada variable

    for var in variables:
        # Prueba de normalidad en ambos grupos
        p_value_trnto = shapiro(df_trnto[var]).pvalue
        p_value_ctrl = shapiro(df_ctrl[var]).pvalue
        # Primero comprobamos si los dos grupos siguen una distribución normal o no usando el test Shapiro-Wilk (para muestras pequeñas)
        
        if p_value_trnto > 0.05 and p_value_ctrl > 0.05:   # Si es normal, usamos t-student
            #Los dos p_values son normales → Usamos T_student
            stat, p_value = ttest_ind(df_trnto[var], df_ctrl[var], equal_var=False)  # Si el p-valor es menor que 0.05, diferencias significativas, Hay diferencias significativas entre los dos grupos
            test_name = 't-test'
        else:
            # Al menos un grupo no es normal → Usamos Mann-Whitney U
            stat, p_value = mannwhitneyu(df_trnto[var], df_ctrl[var], alternative='two-sided')
            test_name = 'Mann-Whitney'

        tests.append(test_name)
        p_values.append(p_value)  # Guardamos el p-valor obtenido

    # Como se hacen muchas pruebas de manera reiterada, aplicamos la corrección FDR de Benjamini-Hochberg para evitar una acumulación de falsos positivos
    reject, p_adjusted, _, _ = multipletests(p_values, method='fdr_bh')

    # Clasificar variables según la corrección FDR
    variablesIncluir = [var for var, r in zip(variables, reject) if r]
    variablesExcluir = [var for var, r in zip(variables, reject) if not r]

    # Guardar los resultados en un DataFrame
    # df_resultados = pd.DataFrame({
    #     'Variable': variables,
    #     'Test aplicado': tests,
    #     'p-valor original': p_values,
    #     'p-valor ajustado (FDR)': p_adjusted,
    #     'Rechazar H0 (FDR)': reject #Esto es si rechazamos la hipótesis nula de que no haya diferencias signitficativas, si es true, las hay, si es false, no las hay
    # })
    print(len(variables),'→',len(variablesIncluir))
    df1 = df1[variablesIncluir]
    return df1

def preprocesadoScale(df1):
    scaler = StandardScaler()
    df_scaled  = scaler.fit_transform(df1)
    df_scaled = pd.DataFrame(df_scaled, columns=df1.columns)
    df_scaled = df_scaled.set_index(df1.index)
    df_scaled2 = df_scaled.copy()
    df_scaled2 = df_scaled2.reset_index()
    df_scaled2['Grupo'] = np.where(df_scaled2['index'].str.contains('A'), 'Tratamiento', 
                np.where(df_scaled2['index'].str.contains('C'), 'Control', 'Otro'))
    df_scaled2 = df_scaled2.set_index('index')
    df_scaled2 = pd.get_dummies(df_scaled2, columns=['Grupo'], drop_first=True)
    df_scaled2['Grupo_Tratamiento'] = df_scaled2['Grupo_Tratamiento'].astype(int)
    return df_scaled2


def RandomForestBootstrap(df_scaled2, n_iteraciones):
    X = df_scaled2.drop(columns=["Grupo_Tratamiento"])  
    y = df_scaled2["Grupo_Tratamiento"]

    max_depth_options = list(range(3, 10))
    min_samples_split_options = list(range(2, 30))
    min_samples_leaf_options = list(range(1, 15))
    n_estimators_options = list(range(1, 200))
    criterions = ['gini', 'entropy']

    resultados = []

    for i in tqdm(range(n_iteraciones), desc="Iterando modelos"):
        # Bootstrap con balanceo
        df_bootstrap = df_scaled2.sample(n=len(df_scaled2), replace=True)

        # Separar clases
        clase_0 = df_bootstrap[df_bootstrap["Grupo_Tratamiento"] == 0]
        clase_1 = df_bootstrap[df_bootstrap["Grupo_Tratamiento"] == 1]

        # Balancear
        if len(clase_0) > len(clase_1):
            clase_1_upsampled = resample(clase_1,
                                         replace=True,
                                         n_samples=len(clase_0),
                                         random_state=None)
            df_balanced = pd.concat([clase_0, clase_1_upsampled])
        else:
            clase_0_upsampled = resample(clase_0,
                                         replace=True,
                                         n_samples=len(clase_1),
                                         random_state=None)
            df_balanced = pd.concat([clase_1, clase_0_upsampled])

        # Mezclar
        df_balanced = df_balanced.sample(frac=1).reset_index(drop=True)

        X_train = df_balanced.drop(columns=["Grupo_Tratamiento"])
        y_train = df_balanced["Grupo_Tratamiento"]

        # OOB set
        oob_mask = ~df_scaled2.index.isin(df_balanced.index)
        X_test = X[oob_mask]
        y_test = y[oob_mask]

        # Hiperparámetros aleatorios
        n_estimators = random.choice(n_estimators_options)
        max_depth = random.choice(max_depth_options)
        min_samples_split = random.choice(min_samples_split_options)
        min_samples_leaf = random.choice(min_samples_leaf_options)
        criterion = random.choice(criterions)

        # Modelo
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random.randint(1, 100000000)
        )
        rf.fit(X_train, y_train)
        y_train_pred = rf.predict(X_train)
        y_test_pred = rf.predict(X_test) if not X_test.empty else []

        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred) if len(y_test_pred) > 0 else np.nan

        important_features = []
        for idx, f in enumerate(X.columns):
            importance = rf.feature_importances_[idx]
            if importance > 0:
                important_features.append((f, importance))

        for feature, importance in important_features:
            resultados.append({
                'Iteration': i,
                'Feature': feature,
                'Importance': importance,
                'Train Accuracy': train_accuracy,
                'Test Accuracy': test_accuracy
            })
    df_resultados_rf = pd.DataFrame(resultados)
    return df_resultados_rf


def elegirVarsImp_rf(df_results_rf,n_iteraciones,thresDesv = 1):
    """
    Elegimos variables más importantes en base a la cantidad de veces que más se repite en los random forests y en base a la importancia más alta
    Las frecuencias e importancias más altas se eligen en base a cuántas desviaciones estandar se alejan por arriba de la media usando z-score.
    """
    temp_df = df_results_rf.groupby('Feature').agg(
        Conteo=('Feature', 'count'),
        Promedio_Importancia=('Importance', 'mean'),
        Promedio_Train_accuracy=('Train Accuracy','mean'),
        Promedio_Test_accuracy=('Test Accuracy','mean')
    ).reset_index()

    # Calcular porcentaje sobre el total
    temp_df['Frecuencia'] = temp_df['Conteo'] / n_iteraciones

    # Ordenar por porcentaje y promedio de importancia
    mostImpFeaturesTree = temp_df.sort_values(by=['Frecuencia', 'Promedio_Importancia'], ascending=False)
    mostImpFeaturesTree = mostImpFeaturesTree[['Feature', 'Conteo', 'Frecuencia','Promedio_Importancia', 'Promedio_Train_accuracy',
        'Promedio_Test_accuracy']]
    mostImpFeaturesTree = mostImpFeaturesTree.reset_index(drop=True)
    frecs_data = mostImpFeaturesTree['Frecuencia']
    z_scores = zscore(frecs_data)
    outliers = frecs_data[z_scores > thresDesv]
    minFrec = outliers.min()
    imp_data = mostImpFeaturesTree['Promedio_Importancia']
    z_scores = zscore(imp_data)
    outliers = imp_data[z_scores > thresDesv]
    minImp = outliers.min()
    mostImpFeaturesTree = mostImpFeaturesTree[(mostImpFeaturesTree['Frecuencia']>=minFrec) & (mostImpFeaturesTree['Promedio_Importancia'] >= minImp)]
    print(len(df_results_rf['Feature'].unique()), 'variables ->', len(mostImpFeaturesTree),'variables')
    return mostImpFeaturesTree

def LogitBootstrap(df,cols, n_iteraciones):
    #cols =mostImpFeaturesTree['Feature'].to_list()
    df2 = df[cols]
    df2 = df2.reset_index()
    df2['Grupo'] = np.where(df2['index'].str.contains('A'), 'Tratamiento', 
                np.where(df2['index'].str.contains('C'), 'Control', 'Otro'))
    df2 = df2.set_index('index')
    df2 = pd.get_dummies(df2, columns=['Grupo'], drop_first=True)
    df2['Grupo_Tratamiento'] = df2['Grupo_Tratamiento'].astype(int)

        # Inicializar lista para guardar los resultados
    resultados = []

    # Número de iteraciones del bootstrap
    n_iter = 10000
    random_state = 42  
    np.random.seed(random_state)

    X_total = df2.drop(columns='Grupo_Tratamiento')
    y_total = df2['Grupo_Tratamiento']
    variables_totales = X_total.columns.tolist()

    for i in tqdm(range(n_iter)):
        # Selección aleatoria de variables
        n_vars = np.random.randint(5, len(variables_totales) + 1)
        vars_seleccionadas = np.random.choice(variables_totales, size=n_vars, replace=False)

        # Subset de variables seleccionadas
        X = X_total[vars_seleccionadas]
        y = y_total

        # Separar por clase
        X_0 = X[y == 0]
        X_1 = X[y == 1]
        y_0 = y[y == 0]
        y_1 = y[y == 1]

        # Balancear (tomar el mínimo de ambas clases para igualar)
        n_samples = min(len(X_0), len(X_1))

        # Bootstrap balanceado por clase
        X_0_boot, y_0_boot = resample(X_0, y_0, replace=True, n_samples=n_samples, random_state=None)
        X_1_boot, y_1_boot = resample(X_1, y_1, replace=True, n_samples=n_samples, random_state=None)

        # Concatenar clases balanceadas
        X_boot = pd.concat([X_0_boot, X_1_boot])
        y_boot = pd.concat([y_0_boot, y_1_boot])

        # Out-of-bag samples: eliminar los índices que fueron usados
        mask = ~X.index.isin(X_boot.index)
        X_oob = X.loc[mask]
        y_oob = y.loc[mask]

        # Escalar
        scaler = StandardScaler()
        X_boot_scaled = scaler.fit_transform(X_boot)
        X_oob_scaled = scaler.transform(X_oob) if not X_oob.empty else None

        # Hiperparámetro aleatorio
        c_value = 10 ** np.random.uniform(-2, 2)

        # Entrenar modelo
        modelo = LogisticRegression(penalty='l1', solver='liblinear', C=c_value, max_iter=1000)
        modelo.fit(X_boot_scaled, y_boot)

        # Evaluar
        if X_oob_scaled is not None and len(y_oob) > 0:
            y_pred_oob = modelo.predict(X_oob_scaled)
            y_pred_boot = modelo.predict(X_boot_scaled)
            r2 = r2_score(y_oob, y_pred_oob)
            acc_oob = accuracy_score(y_oob, y_pred_oob)
            acc_boot = accuracy_score(y_boot, y_pred_boot)
        else:
            r2 = np.nan
            acc_oob = np.nan
            acc_boot = accuracy_score(y_boot, modelo.predict(X_boot_scaled))

        # Guardar resultados
        for var, coef in zip(vars_seleccionadas, modelo.coef_[0]):
            resultados.append({
                'Iteración': i,
                'Variable': var,
                'Coeficiente': coef,
                'C': c_value,
                'R2': r2,
                'Accuracy_Test': acc_oob,
                'Accuracy_Train': acc_boot
            })

    # Convertir a DataFrame final
    df_resultados_log = pd.DataFrame(resultados)
    
    return df_resultados_log

def elegirVarsImp_log(df_resultados_log):
    df_resultados_log2 = df_resultados_log[df_resultados_log['Coeficiente'] != 0]
    temp_df = df_resultados_log2.groupby('Variable').agg(
        Conteo=('Variable', 'count'),
        Promedio_Coeficiente=('Coeficiente', 'mean'),
        Promedio_Train_accuracy=('Accuracy_Train','mean'),
        Promedio_Test_accuracy=('Accuracy_Test','mean')
    ).reset_index()
    temp_df['Importancia (abs)'] = np.abs(temp_df['Promedio_Coeficiente'])
    temp_df = temp_df.sort_values(by=['Importancia (abs)'], ascending=False).drop(columns='Importancia (abs)').reset_index(drop=True)
    print(len(df_resultados_log['Variable'].unique()),'variables', '->', len(df_resultados_log2['Variable'].unique()),'variables')
    return temp_df