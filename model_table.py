from matplotlib import pyplot as plt
import numpy as np
randn = np.random.randn
from matplotlib.font_manager import FontProperties
from pandas import *
import os

def get_num_params(arch,n_layer,int_dim,frac=0):
    if arch=='resbottle':
        num_params = 12 * int_dim + int_dim + \
                        n_layer*(\
                            int_dim * (int_dim//frac) + int_dim//frac + \
                            (int_dim//frac)**2 + int_dim//frac + \
                            (int_dim//frac) * int_dim + int_dim) + \
                        int_dim * 780 + 780 + 2

    elif arch=='resnet' or arch=='mlp':
        multiplier = 1
        if arch == 'resnet':
            multiplier = 2
        num_params = 12 * int_dim + int_dim + \
                        n_layer * multiplier * (int_dim**2 + int_dim) + \
                        int_dim * 780 + 780 + 2

    return num_params

# n_layers_list = np.array([1,2,3,4,5])
# int_dim_list  = np.array([64,128,256,512,1024])
# dim_frac_list = np.array([4,8,16])
# int_dim_bottle= np.array([128,256,512,1024])

n_layers_list = np.array([1,3,5])
int_dim_list  = np.array([64,256,1024])
dim_frac_list = np.array([4,8,16])
int_dim_bottle= np.array([64,256,1024])

names_x_mlp = []
names_y_mlp = []

names_x_resnet = []
names_y_resnet = []

names_x_resbottle = []
names_y_resbottle = []

#for other plot
x_resbottle = []
y_resbottle = []
x_resnet    = []
y_resnet    = []
x_mlp       = []
y_mlp       = []

for i in range(len(int_dim_list)):
    names_x_mlp.append(str(int_dim_list[i]))
    names_x_resnet.append(str(int_dim_list[i]))

for i in range(len(int_dim_bottle)):
    for j in range(len(dim_frac_list)):
        int_dim = int_dim_bottle[i]
        frac    = dim_frac_list[j]

        names_x_resbottle.append(str(int_dim)+' / '+str(frac))

for i in range(len(n_layers_list)):
    names_y_mlp.append('   '+str(n_layers_list[i])+'  ')
    names_y_resnet.append('   '+str(n_layers_list[i])+'  ')
    names_y_resbottle.append('   '+str(n_layers_list[i])+'  ')

vals_mlp       =  np.zeros((3,3))
vals_resnet    = np.zeros((3,3))
vals_resbottle = np.zeros((9,3))

#open the data
result_files = os.listdir('./delta_chi2_T64/')
for file in result_files:
    print('reading file: '+str(file))

    #parse filename for archetecture information
    parse = file[:-4].split('_')

    if 'mlp' not in parse and 'resnet' not in parse and 'densenet' not in parse and 'attention' not in parse:
        arch = 'resbottle'
        # n_layer = int(parse[3])
        # int_dim = int(parse[5])
        # frac    = int(parse[7])
        n_layer = int(parse[1])
        int_dim = int(parse[2])
        frac    = int(parse[3])

    elif 'mlp' in parse:
        arch = 'mlp'
        # n_layer = int(parse[4])
        # int_dim = int(parse[6])
        n_layer = int(parse[1])
        int_dim = int(parse[2])
        frac=0
    elif 'resnet' in parse:
        arch = 'resnet'
        # n_layer = int(parse[4])
        # int_dim = int(parse[6])
        n_layer = int(parse[1])
        int_dim = int(parse[2])
        frac=0
    elif 'densenet' in parse:
        continue
    else:
        print('could not determine architecture')

    num_params = get_num_params(arch,n_layer,int_dim,frac)
    chi2 = np.loadtxt('./delta_chi2_T64/'+str(file))
    chi2_avg = np.mean(chi2)

    if arch=='resbottle':
        idx_x = np.where(int_dim_bottle==int_dim)[0]*3+np.where(dim_frac_list==frac)[0]
        idx_y = np.where(n_layers_list==n_layer)[0]
        vals_resbottle[idx_x,idx_y] = chi2_avg

        x_resbottle.append(num_params)
        y_resbottle.append(chi2_avg)

    elif arch=='mlp':
        idx_x = np.where(int_dim_list==int_dim)[0]#*4+np.where(dim_frac_list==frac)[0]
        idx_y = np.where(n_layers_list==n_layer)[0]
        vals_mlp[idx_x,idx_y] = chi2_avg

        x_mlp.append(num_params)
        y_mlp.append(chi2_avg)

    elif arch=='resnet':
        idx_x = np.where(int_dim_list==int_dim)[0]#*4+np.where(dim_frac_list==frac)[0]
        idx_y = np.where(n_layers_list==n_layer)[0]
        vals_resnet[idx_x,idx_y] = chi2_avg

        x_resnet.append(num_params)
        y_resnet.append(chi2_avg)

arches = ['mlp','resbottle','resnet']
for plot_arch in arches:
    if plot_arch=='resbottle':
        fig = plt.figure(figsize=(9,3))
        vals = vals_resbottle.transpose()
        names_x = names_x_resbottle
        names_y = names_y_resbottle
        x=9
    elif plot_arch=='resnet':
        fig = plt.figure(figsize=(3,3))
        vals=vals_resnet.transpose()
        names_x = names_x_resnet
        names_y = names_y_resnet
        x=3
    elif plot_arch=='mlp':
        fig = plt.figure(figsize=(3,3))
        vals=vals_mlp.transpose()
        names_x = names_x_mlp
        names_y = names_y_mlp
        x=3

    ax = fig.add_subplot(111, frameon=False, xticks=[], yticks=[])

    img = plt.imshow(vals, cmap="cividis",vmin=0, vmax=5)  #choosing color map here

    plt.colorbar(location="bottom",label='Average $\Delta\chi^2$')#.ax.tick_params(labelsize=16)

    img.set_visible(False)
    the_table = plt.table(
        #cellText=vals,
        rowLabels = names_y, 
        colLabels = names_x, 
        #colWidths = [0.068]*vals.shape[1],
        rowLoc='center',
        loc = 'center',
        cellLoc = 'center',
        cellColours = img.to_rgba(vals))

    for cell in the_table._cells:
        the_table._cells[cell].set(height=0.3)#,width=0.068)
        for i in np.arange(x):
           the_table._cells[0,i].get_text().set_rotation(50)

      # if row==0:
      #   cell.set_linewidth(0)
      #   cell.set_height(0.1)

    #custom heading titles - new portion
    # print([0.068]*vals.shape[1])
    # width=5
    # height=3
    # col_width=0.068

    # hoffset=0.0 #find this number from trial and error
    # voffset=0.98 #find this number from trial and error
    # line_fac=0.5 #controls the length of the dividing line
    # count=0

        # #add a dividing line
        # ax.annotate('', xy=(hoffset+(count+0.5)*col_width,voffset), 
        #     xytext=(hoffset+(count+0.5)*col_width+line_fac/width,voffset+line_fac/height),
        #     xycoords='axes fraction', arrowprops={'arrowstyle':'-'})


    # for (row, col), cell in the_table.get_celld().items():
    #   if (row == 0) or (col == -1):
    #     cell.set_text_props(fontproperties=FontProperties(weight='bold'))
    #     cell.set_width(0.1)



    # for (row, col), c in the_table.get_celld().values():
    #     if (row == 0) or (col == -1):
    #         c.visible_edges = 'horizontal'


    #ax.text(0.5, 0.03, 'Tension Metric 1: Parameter Difference', ha='center', size=16)
    #ax.text(0.5, 0.95, 'Tension Metric 2: Parameter Differences in Update Form', ha='center', size=16)
    #ax.text(0.9, 0.2,  chains_description, ha='left',linespacing = 3, size=12)

    plt.savefig('model_table_'+str(plot_arch)+'_T64.pdf',bbox_inches='tight')

# resbottle_idxs = np.argsort(np.array(x_resbottle))
# resnet_idxs    = np.argsort(np.array(x_resnet))
# mlp_idxs       = np.argsort(np.array(x_mlp))
# attention      = np.mean(np.load('delta_chi2_data/attention_transformer_test_500_epochs_deltachi2.npy'))
# f1 = plt.figure()
# plt.plot(np.array(x_resbottle)[resbottle_idxs],np.array(y_resbottle)[resbottle_idxs],c='r',label='ResBottle',marker='o',linestyle='solid')
# plt.plot(np.array(x_resnet)[resnet_idxs],np.array(y_resnet)[resnet_idxs],c='g',label='ResNet',marker='v',linestyle='dashed')
# plt.plot(np.array(x_mlp)[mlp_idxs],np.array(y_mlp)[mlp_idxs],c='b',label='MLP',marker='^',linestyle='dotted')
# #plt.plot([1646490],attention,c='c',marker='D',label='ResNet+Transformer',linestyle='None')
# #plt.ylim([0,0.1])
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel('Number of parameters')
# plt.ylabel('Mean $\chi^2$ between emulator and cocoa')
# plt.legend()
# f1.savefig('avg_chi2_v_n_params_new.pdf')

