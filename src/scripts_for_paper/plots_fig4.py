from os import name
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

if __name__ == "__main__":
    """
    find a way to generate csv files from model prediction stats
    """
    
    # GRAPH LACZ VS IL34 -> vol meta / day (meme souris sur plusieurs jours)
    # fig 4 pannel A
    
    
    data = pd.read_csv('/home/edgar/Documents/Projects/deepmeta_v2/lacz_vs_il34_cleaned.csv')
    df = pd.DataFrame(data)

    print(df)
    sns.set(rc={"figure.figsize":(8, 7)})
    sns.set_theme(style="whitegrid")
    ax = sns.catplot(x="day", y="vol_m", hue="mutation", data=df, kind="bar", ci=None, palette="rocket", height=6, aspect=1.5, dodge=False, legend_out=False)
    ax.set(xlabel="Day", ylabel="Volume metastases (mm3)")
    plt.tight_layout()
    plt.show()
    
    
    # GRAPH 1 LIGNE 1 META -> NB meta / VOL -> 1 souris de chaque groupe
    # fig 4 pannel B
    
    # data = pd.read_csv('/home/edgar/Documents/Projects/deepmeta_v2/metaid.csv')
    # df = pd.DataFrame(data)


    # # print(df)
    # bins = [0.001, 1, 5, 10, 50 ,100, 250, 500]
    # new_df = df.groupby("mutation")["vol"].value_counts(bins=bins)

    # print(new_df)
    # df2 = new_df.reset_index(name="occ").sort_values("vol")
    # print(df2.sort_values("vol"))

    # sns.set(rc={"figure.figsize":(8, 7)})
    # sns.set_theme(style="whitegrid")
    # ax = sns.barplot(x="vol", y="occ", hue="mutation", data=df2, ci=None, palette="rocket")
    # ax.set(xlabel="Metastases volume", ylabel="Metastases number")
    # ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right", size=10)
    # plt.tight_layout()
    # plt.show()

    # GRAPH 1 LIGNE 1 META -> NB meta / VOL
    # fig 4 pannel C
    
    # data = pd.read_csv('/home/edgar/Documents/Projects/DeepMeta/iso_mice.csv')
    # df = pd.DataFrame(data)


    # # print(df)
    # bins = [0.0, 1, 5, 10, 50 ,100, 250, 500, 2000]
    # new_df = df.groupby("Mice")["vol"].value_counts(bins=bins)

    # print(new_df)
    # df2 = new_df.reset_index(name="occ").sort_values("vol")
    # print(df2.sort_values("vol"))
    # sns.set(rc={"figure.figsize":(8, 7)})
    # sns.set_theme(style="whitegrid")
    # ax = sns.barplot(x="vol", y="occ", hue="Mice", data=df2, ci=None, palette="rocket")
    # ax.set(xlabel="Metastases volume", ylabel="Metastases number") 
    # ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right", size=10)
    # plt.tight_layout()
    # plt.show()


