import numpy as np
import copy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pyrepo_mcda import normalizations as norms
from pyrepo_mcda.additions import rank_preferences
from vassp import VASSP

from pyrepo_mcda import weighting_methods as mcda_weights
from pyrepo_mcda import correlations as corrs

from matplotlib.pyplot import cm



def main():

    # Symbols of Countries
    coun_names = pd.read_csv('./DATASET/SHARES_2022_POPULATION.csv')
    country_names = list(coun_names['Country'])
    

    df_data = pd.read_csv('./DATASET/SHARES_2022_POPULATION.csv', index_col='Country')
    matrix = df_data.to_numpy()
    types = np.ones(matrix.shape[1])
    weights = mcda_weights.critic_weighting(matrix)

    coeffs = np.arange(0.0, 1.1, 0.1)

    df_rank_v = pd.DataFrame(index = df_data.index)
    df_pref_v = pd.DataFrame(index = df_data.index)

    for coeff in coeffs:

        s_coeff = np.ones(matrix.shape[1]) * coeff
        
        # SSP-VIKOR method
        vassp = VASSP()

        pref_vassp = vassp(matrix, weights, types, s_coeff = s_coeff)
        rank_vassp = rank_preferences(pref_vassp, reverse = False)

        df_pref_v[str(coeff)] = pref_vassp
        df_rank_v[str(coeff)] = rank_vassp


    #
    # ==================================================================================
    # plot figure with sensitivity analysis

    plt.figure(figsize = (10, 6))
    for k in range(df_rank_v.shape[0]):
        plt.plot(coeffs, df_rank_v.iloc[k, :], '.-', linewidth = 3)

        ax = plt.gca()
        y_min, y_max = ax.get_ylim()
        x_min, x_max = ax.get_xlim()
        
        plt.annotate(country_names[k], (x_max, df_rank_v.iloc[k, -1]),
                        fontsize = 14, style='italic',
                        horizontalalignment='left')

    # plt.xlabel(r'$s$' + ' coefficient value', fontsize = 14)
    plt.xlabel('Sustainability coefficient', fontsize = 14)
    plt.ylabel("Rank", fontsize = 14)
    
    plt.xticks(coeffs, fontsize = 14)
    plt.yticks(ticks=np.arange(1, len(country_names) + 1, 1), fontsize = 14)

    plt.gca().invert_yaxis()
    plt.xlim(x_min, x_max + 0.2)
    plt.grid(True, linestyle = ':')
    plt.title('All criteria compensation reduction', fontsize = 14)
    plt.tight_layout()
    plt.savefig('results/sust_coeff' + '.pdf')
    plt.show()
    

    df_rank_v = df_rank_v.rename_axis('Country')
    df_rank_v.to_csv('./results/df_rankings_ssp_vikor.csv')

    df_pref_v = df_pref_v.rename_axis('Country')
    df_pref_v.to_csv('./results/df_pref_ssp_vikor.csv')


    # sustainability coefficient from matrix calculated based on standard deviation from normalized matrix
    n_matrix = norms.minmax_normalization(matrix, types)
    s = np.sqrt(np.sum(np.square(np.mean(n_matrix, axis = 0) - n_matrix), axis = 0) / n_matrix.shape[0])


    # analysis with sustainability coefficient modification
    model = [
        [0, 1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12, 13, 14],
        [15, 16, 17],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    ]

    results_pref = pd.DataFrame(index = df_data.index)
    results_rank = pd.DataFrame(index = df_data.index)

    #
    # analysis performed for table
    for el, mod in enumerate(model):
        new_s = np.zeros(matrix.shape[1])
        new_s[mod] = s[mod]

        pref_vassp = vassp(matrix, weights, types, s_coeff = new_s)
        rank_vassp = rank_preferences(pref_vassp, reverse = False)
        results_pref['SSP-VIKOR ' + r'$G_{' + str(el + 1) + '}$'] = pref_vassp
        results_rank['SSP-VIKOR ' + r'$G_{' + str(el + 1) + '}$'] = rank_vassp

    results_pref = results_pref.rename_axis('Country')
    results_rank = results_rank.rename_axis('Country')
    results_pref.to_csv('./results/df_pref_G' + '.csv')
    results_rank.to_csv('./results/df_rank_G' + '.csv')

    
    # analysis performed for different sustainability coefficients
    sust_coeff = np.arange(0, 1.1, 0.1)

    for el, mod in enumerate(model):
        results_pref = pd.DataFrame(index=country_names)
        results_rank = pd.DataFrame(index=country_names)

        for sc in sust_coeff:

            s = np.zeros(matrix.shape[1])
            s[mod] = sc

            pref = vassp(matrix, weights, types, s_coeff = s)
            rank = rank_preferences(pref, reverse = False)

            results_pref[str(sc)] = pref
            results_rank[str(sc)] = rank


        results_pref = results_pref.rename_axis('Country')
        results_rank = results_rank.rename_axis('Country')
        results_pref.to_csv('./results/df_pref_sust_G' + str(el + 1) + '.csv')
        results_rank.to_csv('./results/df_rank_sust_G' + str(el + 1) + '.csv')

        # plot results of analysis with sustainabiblity coefficient modification
        ticks = np.arange(1, matrix.shape[0])

        x1 = np.arange(0, len(sust_coeff))

        plt.figure(figsize = (10, 6))
        for i in range(results_rank.shape[0]):
            plt.plot(x1, results_rank.iloc[i, :], '.-', linewidth = 3)
            ax = plt.gca()
            y_min, y_max = ax.get_ylim()
            x_min, x_max = ax.get_xlim()
            plt.annotate(country_names[i], (x_max, results_rank.iloc[i, -1]),
                            fontsize = 14, style='italic',
                            horizontalalignment='left')

        plt.xlabel("Sustainability coeffcient", fontsize = 14)
        plt.ylabel("Rank", fontsize = 14)
        plt.xticks(x1, np.round(sust_coeff, 2), fontsize = 14)
        plt.yticks(ticks, fontsize = 14)
        plt.xlim(x_min - 0.2, x_max + 2.1)
        plt.gca().invert_yaxis()
        
        plt.grid(True, linestyle = ':')
        if el < 4:
            plt.title(r'$G_{' + str(el + 1) + '}$')
        else:
            plt.title('All criteria')
        plt.tight_layout()
        plt.savefig('./results/rankings_sust_G' + str(el + 1) + '.pdf')
        plt.show()
    



if __name__ == '__main__':
    main()