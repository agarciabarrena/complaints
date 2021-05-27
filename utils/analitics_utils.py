import shap
import pandas as pd


def show_sha_analysis(model, df_train: pd.DataFrame):
    explainer = shap.Explainer(model)
    shap_values = explainer(df_train)
    shap.plots.bar(shap_values)  # Features importance plot
    shap.plots.beeswarm(shap_values)
    shap_interaction_values = explainer.shap_interaction_values(df_train)
    shap.summary_plot(shap_interaction_values, df_train)


def plot_groups(df: pd.DataFrame, col_check: str='clase'):
    cols_2_analyze = [c for c in df.columns if c != col_check]
    for col in cols_2_analyze:
        print(col)
        temp = df[[col, col_check]].copy()
        if temp[col].dtype == 'O':
            pvt = pd.pivot_table(temp,
                                 columns=col_check,
                                 index=col,
                                 aggfunc=len,
                                 fill_value=0)
            pvt['total'] = pvt['h'] + pvt['l']
            pvt.sort_values('total',ascending=False, inplace=True)
            pvt['pct'] = pvt['h'] / pvt['l']
            print(pvt[['h', 'l', 'pct']])
        else:
            for c in temp[col_check].unique():
                print(c)
                print(df[df[col_check] == c][col].describe().round(2))
        print('\n\n')
    return None