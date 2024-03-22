from scipy.stats import friedmanchisquare
import scikit_posthocs as sp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
#import tikzplotlib
from matplotlib.colors import LogNorm, BoundaryNorm, ListedColormap

plt.style.use("ggplot")
#rf = [5,5.5,4,6,6,5,4,5.5,5,6,6,3,6]
# 16, 32
#rf = [5,5,4,6,6,5,4,5,5,6,6,3,6,6,2,4,2,6,7,5,7,6,6,5,2,2,6,2,3,5,4,7,2,7,4,7,7,7,3,6,5,4,2,6,3,4,7,5,6,4,2,4,5,3,3,6,6,4,3,6,4,7,7,7,2]
rf =[4,5,4,6,6,6,4,6,4,6,7,4,6,4,3,4,4,6,6,4,7,5,6,7,5,3,4,5,6,4,3,4,2,7,5,6,5,4,4,4,4,4,2,6,4,2,7,5,6,5,4,6,4,4,4,4,3,3,4,7,5,6,5,3,5]
print(np.average(rf))
#ssf = [2,1.5,2.5,3,2,1,2,1,1.5,4,1,2,2]
#ssf = [1,1,2,3,	2,	1,	1,	1,	1,	4,	1,	1,	1,	1,	1,	2,	3,	2,	1,	1,	1,	1,	4,	1,	1,	1,	1,	1,	2,	3,	2,	1,	1,	1,	1,	4,	1,	1,	1,	1,	1,	2,	3,	2,	1,	1,	1,	1,	4,	1,	1,	1,	1,	1,	2,	3,	2	,1,	1,	1,	1,	4,	1,	1,	1]
ssf = [2,1,2,3,2,1,2,1,1,4,1,2,2,2,1,2,2,2,1,2,1,1,4,1,2,2,2,1,2,3,2,1,2,1,1,5,2,2,2,2,1,2,3,2,1,3,1,1,4,1,2,2,2,1,2,3,2,1,2,1,1,4,1,2,2]
print(np.average(ssf))
#leaf = [1,1.5,1,5.5,1,3,1,4.5,3,2,3,1,1]
#l1lr = [2,2,1,3,1,2,2,4,3,2,2,2,2,2,2,1,3,1,2,2,3,2,2,2,3,2,2,2,1,3,1,2,2,3,2,2,2,2,3,2,2,1,3,1,2,2,3,2,2,2,3,2,2,2,1,3,1,2,2,3,2,2,3,2,2]
l1lr = [1,1,1,4,1,3,1,4,4,2,3,1,1,1,1,1,4,1,3,1,3,4,2,3,1,1,1,1,1,6,1,3,1,3,4,2,4,1,1,1,1,1,4,1,3,1,3,4,2,3,1,1,1,1,1,5,1,4,1,3,3,2,4,1,1]
print(np.average(l1lr))
#ir = [5,3,6,2,3,5,5,4.5,1.5,3,5,7,3.5]
#ir = [4,2,6,	2,	5,	7,	5,	5,	1,	3,	3,	6,	2,	4,	5,	6,	5,	4,	6,	6,	4,	3,	3,	3,	4,	2,	4,	5,	5,	1,	4,	6,	5,	6,	3,	3,	3,	4,	5,	4,	4,	6,	5,	5,	6,	5,	6,	3,	3,	6,	6,	2,	4,	3,	6,	2,	5,	5,	5,	4,	5,	3,	2,	4,	2]
ir = [5,3,6,2,3,5,5,4,1,3,5,7,3,6,5,5,4,5,5,6,5,3,2,4,6,3,6,1,4,1,5,5,5,6,3,3,1,6,4,6,5,5,6,5,5,5,6,3,3,4,6,3,5,4,6,2,6,5,5,4,3,3,3,6,3]
print(np.average(ir))

#drep = [7,4,5,4,5,4,6,3,4,5,4,6,3.5]
#drep= [7,4,4,5,3,4,6,3,4,5,7,5,2,7,2,5,5,4,4,4,4,4,5,7,5,5,7,2,6,6,6,5,6,5,4,5,6,5,1,7,2,4,6,3,7,6,4,4,5,7,4,4,7,3,5,5,3,7,6,7,3,5,6,5,2]
drep = [6,4,5,4,5,4,6,3,3,5,4,6,3,5,3,5,2,4,4,5,3,2,5,5,4,5,5,4,4,4,6,6,6,5,2,4,7,5,2,5,3,6,4,3,6,6,4,2,5,7,5,4,6,1,4,5,4,6,6,5,2,5,7,5,3]
print(np.average(drep))
#ccp =[6,6,7,7,7,7,7,2,6,7,6,5,7]
#ccp = [6,5,7,7,7,3,7,2,7,7,5,7,7,5,6,7,7,7,3,7,2,7,7,6,7,7,5,6,7,6,7,3,7,2,7,6,5,6,7,5,6,7,7,7,4,7,2,7,7,4,7,7,6,6,7,7,7,3,6,2,7,6,5,6,7]
ccp = [7,6,7,7,7,7,7,2,6,7,6,5,7,7,6,7,7,7,7,7,2,5,7,5,7,7,7,6,7,7,7,7,7,2,6,7,6,7,7,7,6,7,7,7,7,7,2,6,7,6,7,7,7,6,7,7,7,7,7,2,6,7,6,7,7]
print(np.average(ccp))
#gr = [3,7,2.5,1,4,2,3,7,7,1,2,3,5]
#lr = [3,7,3,1,4,6,3,7,6,1,4,4,5,3,7,3,1,3,5,3,6,5,1,4,6,5,3,7,3,1,3,4,4,4,6,1,4,3,5,3,7,3,1,4,5,3,5,5,1,3,5,4,3,7,3,1,4,6,4,5,6,1,4,3,6]
lr = [3,7,2,1,4,2,3,7,7,1,2,3,5,3,7,2,1,3,2,3,6,7,1,2,3,5,3,7,2,1,4,2,4,4,7,1,3,3,4,3,7,2,1,4,2,4,5,7,1,2,3,4,3,7,2,1,5,2,3,5,7,1,2,3,5]
print(np.average(lr))

res = friedmanchisquare(rf,ssf,l1lr,ir,drep,ccp, lr)

print(res.statistic)

print('p value friedman')
print(res.pvalue)

data = np.array([rf,ccp,drep, ir,lr,l1lr,ssf])

nemen_res = sp.posthoc_nemenyi_friedman(data.T)

co = sp.posthoc_conover(data)
#co[:] = np.tril(co.values, k=-1)
print(type(co))
print(nemen_res)
print('conover: ')
print(co)
heatmap_args = {'cmap': ['1', '#fb6a4a',  '#08306b',  '#4292c6', '#c6dbef'],
                'linewidths': 0.25,
                'linecolor': '0.5',
                'clip_on': False,
                'square': True,
                'cbar_ax_bbox': [1, 0.35, 0.04, 0.3],
               }
cmap = ['1', '#fb6a4a',  '#08306b',  '#4292c6', '#c6dbef']
heatmap_args = {'cmap': cmap, 'linewidths': 0.25, 'linecolor': '0.5', 'clip_on': False, 'square': True, 'cbar_ax_bbox': [0.83, 0.35, 0.04, 0.3]}

#co.rename(columns={'1': 'newName1', '2': 'newName2'}, inplace=True)#co = pd.DataFrame({'Column1': co[:, 0], 'Column2': co[:, 1],'Column3': co[:, 2]})
co.columns = ['RF','CCP','DREP','IE','LR','LR+L1','SSF']
co.index = ['RF','CCP','DREP','IE','LR','LR+L1','SSF']
print(co)
#heatmap_args = {'linewidths': 0.25, 'linecolor': '0.5', 'clip_on': False, 'square': True, 'cbar_ax_bbox': [0.80, 0.35, 0.04, 0.3]}
#sp.sign_plot(co)
plt.xticks(fontsize = 28)
plt.yticks(fontsize = 28)
sp.sign_plot(co,**heatmap_args)

#plt.set_title('Significance plot', fontsize=14)
#plt.show()
plt.savefig('conover_32.png', bbox_inches= 'tight')
'''
from scipy.stats import pearsonr
np.random.seed(42)
data = pd.DataFrame(np.random.randn(100, 4), columns=list('ABCD'))

# Calculate the correlation matrix using Pearson correlation coefficients
corr_matrix = data.corr(method='pearson')

# Calculate the p-values for the correlation coefficients
pvalues = round(data.corr(method=lambda x, y: pearsonr(x, y)[1]), 4)

# Generate the heatmap of the correlation matrix
mask = np.triu(np.ones_like(co, dtype=bool))
sns.set(style='white')
fig, ax = plt.subplots(figsize=(10, 8))

my_colors = ['yellow', 'red', 'blue', 'red']
bounds = [0, 0.001, 0.01, 0.05, 1.0]
my_cmap = ListedColormap(my_colors)
my_norm = BoundaryNorm(bounds, ncolors=len(my_colors))
sns.heatmap(co, annot=True, fmt='.2f', cmap=my_cmap, mask=mask, cbar_kws={'shrink': 0.8},
            ax=ax, vmin=0, vmax=1, center=0.05, norm=my_norm)


plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()
plt.close()

ranking_data = {
  "dataset": ['adult', 'adult', 'aloi'],
  "method": ['RF', 'SSF','RF'],
}

f, (ax) = plt.subplots(1, 1, figsize=(12, 4))
f.suptitle('T', fontsize=14)
ranking = [rf,ssf,l1lr,ir,drep,ccp,lr]
#ranking = pd.DataFrame(ranking)
#print(ranking)
sns.boxplot(data=ranking)
ax.set_xlabel("Method",size = 12,alpha=0.8)
ax.set_ylabel("Rank",size = 12,alpha=0.8)
plt.show()
'''