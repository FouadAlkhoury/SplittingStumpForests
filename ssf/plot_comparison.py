import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib as mpl
import matplotlib

print(matplotlib.__version__)
# ## color: acc, shape: method

# data = np.genfromtxt('accvscompression.maxdepth5', skip_header=1,
# dtype=[('s1','S10'),('f1','f8'),('f2','f8'),('f3','f8'),('mystring','S4')])
# print(data[0])

# marker = {b'RF': 'o', b'SSF': '*', b'CCP': '^', b'GR': 'D'}
# markersize = 7
# legendmarkers = [
#     mlines.Line2D([], [], color='black', marker='o', markersize=markersize,
#                           label='RF', linestyle='none'),
#     mlines.Line2D([], [], color='black', marker='*', markersize=markersize,
#                           label='SSF', linestyle='none'),
#     mlines.Line2D([], [], color='black', marker='^', markersize=markersize,
#                           label='CCP', linestyle='none'),
#     mlines.Line2D([], [], color='black', marker='D', markersize=markersize,
#                           label='GR', linestyle='none'),
# ]

# vmin = np.min([x[1] for x in data])
# vmax = np.max([x[1] for x in data])

# for x in data:
#     artist = plt.scatter(
#         x=x[3],
#         y=x[2],
#         c=x[1],
#         s=5*markersize,
#         marker=marker[x[4]],
#         vmin=vmin, vmax=vmax,
#     )

# plt.colorbar()
# plt.xlabel('Inference Time [ms]')
# plt.ylabel('Compression Ratio')
# plt.legend(handles=legendmarkers, loc="lower right")
# plt.savefig('maxdepth5_accuracycolor.pdf')
# plt.show()


## color: dataset, shape: method

data = np.genfromtxt('dataset.csv', skip_header=1,
dtype=[('s1','S10'),('f1','f8'),('f2','f8'),('f3','f8'),('mystring','S4')])
print(data[0])

marker = {b'RF': 'o', b'CCP': '^', b'DREP':'<', b'IE':'v',b'LR': 'D', b'LRL1': 's', b'SSF': '*'}
markersize = 7
legendmarkers = [
    mlines.Line2D([], [], color='black', marker='o', markersize=markersize,
                          label='RF', linestyle='none'),
    mlines.Line2D([], [], color='black', marker='^', markersize=markersize,
                          label='CCP', linestyle='none'),
    mlines.Line2D([], [], color='black', marker='<', markersize=markersize,
                          label='DREP', linestyle='none'),
    mlines.Line2D([], [], color='black', marker='v', markersize=markersize,
                          label='IE', linestyle='none'),
    mlines.Line2D([], [], color='black', marker='D', markersize=markersize,
                          label='LR', linestyle='none'),
    mlines.Line2D([], [], color='black', marker='s', markersize=markersize,
                          label='LRL1', linestyle='none'),
    mlines.Line2D([], [], color='black', marker='*', markersize=markersize,
                          label='SSF', linestyle='none'),
]

color = {
    b'adult': 0,
    b'aloi': 1,
    b'bank': 2,
    b'credit': 3,
    b'drybean': 4,
    b'letter': 5,
    b'magic': 6,
    b'rice': 7,
    b'room': 8,
    b'shoppers': 9,
    b'spambase': 10,
    b'statlog': 11,
    b'waveform': 12,
}

vmin = 0
vmax = 12

amin = np.min([x[1] for x in data])
amax = np.max([x[1] for x in data])


# fiddle with colors. taken from
#https://stackoverflow.com/questions/14777066/matplotlib-discrete-colorbar
cmap = plt.cm.tab20b  # define the colormap
# extract all colors from the map
cmaplist = [cmap(i) for i in range(cmap.N)]
# ensure that we have exactly as many colors as datasets
cmaplist = cmaplist[:len(color)]
# force the first color entry to be grey
# cmaplist[0] = (.5, .5, .5, 1.0)

# create the new map
cmap = mpl.colors.LinearSegmentedColormap.from_list(
    'Custom cmap', cmaplist, len(color))

# # define the bins and normalize
# bounds = np.linspace(0, 20, 21)
# norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

#fig = plt.figure(layout='constrained', figsize=(10, 4))
fig = plt.figure(layout='constrained', figsize=(10, 4))

#plt.xticks(fontsize = 16)
#plt.yticks(fontsize = 16)
axs = fig.subplots(1, 2, sharey=False)
plt.setp(axs[0].get_xticklabels(), fontsize=16)
plt.setp(axs[0].get_yticklabels(), fontsize=16)
plt.setp(axs[1].get_xticklabels(), fontsize=16)
for x in data:
    thingie = axs[0].scatter(
        x=x[2],
        y=x[3],
        c=color[x[0]],
        #s=20+200*(x[1]-amin)/(amax-amin),
        marker=marker[x[4]],
        vmin=vmin, vmax=vmax,
        cmap=cmap
    )
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
axs[0].set_xlabel('Compression Ratio', fontsize = 18)
axs[0].set_ylabel('Inference Time [ms]', fontsize = 18)


cbar = fig.colorbar(thingie)
cbar.ax.get_yaxis().set_ticks(12*np.array(range(len(color)))/13+0.5)
cbar.ax.get_yaxis().set_ticklabels([b.decode('UTF-8') for b in color.keys()], fontsize=16)

for x in data:
    thingie = axs[1].scatter(
        x=x[2],
        y=x[1],
        c=color[x[0]],
        #s=20+200*(x[1]-amin)/(amax-amin),
        marker=marker[x[4]],
        vmin=vmin, vmax=vmax,
        cmap=cmap
    )
axs[1].yaxis.tick_right()
axs[1].yaxis.set_label_position("right")
axs[1].set_xlabel('Compression Ratio', fontsize = 18)
axs[1].set_ylabel('Accuracy', fontsize = 18)


axs[0].legend(handles=legendmarkers, loc=(0,0.56), fontsize = 10)
#axs[0].legend(handles=legendmarkers, loc="upper left")
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
plt.savefig('maxdepth5_datasetcolor.png')
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
plt.show()