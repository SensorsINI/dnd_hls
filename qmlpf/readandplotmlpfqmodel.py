import csv  
from matplotlib import pyplot as plt  
from datetime import datetime 
import matplotlib
from matplotlib.pyplot import MultipleLocator
import matplotlib.ticker as ticker
import numpy as np

import os,sys

from numpy.core.fromnumeric import sort

rootpath= 'D://' # adjusted according to your path
npyfolder = sys.argv[1] # the folder where the npy data are
files = os.listdir(npyfolder)
targetfolder = sys.argv[2] # the folder for storing result images
application = sys.argv[3] # incicating which dataset/application is used
# if len(sys.argv) < 3:
#     return "Parameters not enough. three parameters needed: npyfolder, targetfolder, application"
filtername = sys.argv[4]
plotflag = sys.argv[5]

if not os.path.exists(targetfolder):
    os.makedirs(rootpath+targetfolder)

methods = []

thrs = []
tprs = []
tnrs = []
brs = []
fprs = []
aucs = []

colors = [['black'],['blue'],['green'],['purple'],['cyan'],['red'],['darkorange'],['plum'],['goldenrod'],['darkblue'],['darkgreen'],['darkred'],['grey'],['yellow'],['olive'],['pink'],['brown']]

matplotlib.rcParams.update({'font.size': 14})
# Say, "the default sans-serif font is COMIC SANS"
matplotlib.rcParams['font.sans-serif'] = "Arial"
# Then, "ALWAYS use sans-serif fonts"
matplotlib.rcParams['font.family'] = "sans-serif"
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

linestyle_str = [
     ('solid', 'solid'),      # Same as (0, ()) or '-'
     ('dotted', 'dotted'),    # Same as (0, (1, 1)) or '.'
     ('dashed', 'dashed'),    # Same as '--'
     ('dashdot', 'dashdot')]  # Same as '-.'


fig=plt.figure() 
cc = 0 
for filename in files:
    # print(filename)
    if 'npy' in filename and application in filename and filtername in filename:
        print(filename)
        arr = np.load(npyfolder + '/' + filename)
        # if 'thr' in filename:
        if True:
            print(arr)
            # print(np.array(arr).shape)
            # tmptprarr = arr[0]
            # tmpfprarr = arr[1]
            # # print(tprarr.shape)
            # # print(fprarr.shape)
            # tprarr = tmptprarr[:,0]
            # fprarr = tmpfprarr[:,0]
            tprarr = np.array(arr[1])
            fprarr = np.array(arr[0])


            sorted_indices = np.argsort(fprarr, axis=0)
            print(sorted_indices)
            sorted_fprarr = fprarr[sorted_indices]
            sorted_tprarr = tprarr[sorted_indices]
            # if 'STCF' in filename or 'mlp' in filename or 'MLP' in:
            if 'ONF' in filename:
                sorted_fprarr = np.concatenate(([0],sorted_fprarr))
                sorted_tprarr = np.concatenate(([0],sorted_tprarr))
            else:
                sorted_fprarr = np.concatenate(([0],sorted_fprarr))
                sorted_tprarr = np.concatenate(([0],sorted_tprarr))
                sorted_fprarr = np.concatenate((sorted_fprarr,[1]))
                sorted_tprarr = np.concatenate((sorted_tprarr,[1]))

            
        # if 'TI' in filename:
        #     # print(arr.shape)
        #     sorted_fprarr = arr[0]
        #     sorted_tprarr = arr[1]
        # auc = arr[2]
        from sklearn.metrics import auc
        aucvalue = auc(sorted_fprarr,sorted_tprarr)
        print('AUC:%.4f'%aucvalue)

        

        elements = filename.split('.')[0].split('_')

        methodpre = elements[0]
        if 'MLP' in methodpre:
            # methodpre = 'MLP'
            methodpost = '(' + elements[-1].replace(application, '') + ')'

        elif elements[-1] == application:
            methodpost = elements[-2]
        elif 'num.npy' in filename:
            # methodpost = '(' + elements[-1].replace(application, '') + ')'
            if plotflag == 't':
                tmp = elements[-1].replace(application, '')
                methodpost = '(ΔT=' + tmp.split('num')[0] + ')'
            else:
                continue
        else:
            if plotflag == 't':
                continue
            else:
                methodpost = '(ψ=' + elements[-1].split('num')[1] + ')'

        # print(methodpre, fprarr,tprarr)
        
        roundauc = '%.4f' % aucvalue
        method = methodpre + methodpost +  ',AUC=' + str(roundauc)

        # method = methodpre + ',' + methodpost + ',AUC=' + str(roundauc)
        
        curcolor = colors[cc][0]
        ls = 'solid'
        # if '2x' in filename:
        #     ls = 'solid'
        if 'oti' in filename:
            ls = 'dashed'
        if 'H0' in filename:
            ls = 'dotted'
        if 'DH' in filename:
            ls = 'dashdot'
        
            
        
        # plt.scatter(x=fprarr,y=tprarr,color=curcolor)
        # plt.plot(fprarr,tprarr,label=method,color=curcolor)
        plt.scatter(x=sorted_fprarr,y=sorted_tprarr,color=curcolor)
        plt.plot(sorted_fprarr,sorted_tprarr,linestyle=ls,label=method,color=curcolor)
        cc += 1

plt.legend(loc='lower right')
 
plt.title('Receiver Operating Characteristic')
plt.plot([(0,0),(1,1)],'r--')
plt.xlim([-0.01,1.01])
plt.ylim([-0.01,1.05])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')

ax=plt.gca()

tick_spacing = 0.05
xtick_spacing = 0.1
ax.xaxis.set_major_locator(ticker.MultipleLocator(xtick_spacing))
ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

x1, y1 = 0,1
x2, y2 = -0.05, 1.1

plt.plot([x1, x2],[y1,y2], "*", color='r')

plt.annotate(r'$Ideal Point$', xy=(x1,y1),xycoords='data',xytext=(x2,y2),textcoords='data',
             fontsize=16,arrowprops=dict(arrowstyle='->',connectionstyle='arc3,rad=0.3'))


plt.savefig(targetfolder + '/' + filtername + application + plotflag + 'AllROC.pdf')  
