# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 23:06:47 2023

@author: hussain
"""

import matplotlib.pyplot as plt

names = ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person']

fig = plt.figure(figsize=(15, 6))
gs = fig.add_gridspec(1, 4, hspace=0, wspace=0)
ax1, ax2, ax3, ax4 = gs.subplots(sharex='col', sharey='row')

values1 = [379, 255, 285, 184, 201, 295, 449]
bars1 = ax1.bar(names, values1, color='grey')
for bar in bars1:
   height = bar.get_height()
   ax1.annotate(f'{height}', xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3),
   textcoords="offset points", ha='center', va='bottom')
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=90)

values2 = [389, 457, 346, 135, 324, 288, 405]
bars2 = ax2.bar(names, values2, color='tan')
for bar in bars2:
   height = bar.get_height()
   ax2.annotate(f'{height}', xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3),
   textcoords="offset points", ha='center', va='bottom')
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=90)

values3 = [189, 202, 182, 186, 199, 280, 432]
bars3 = ax3.bar(names, values3, color='darkcyan')
for bar in bars3:
   height = bar.get_height()
   ax3.annotate(f'{height}', xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3),
   textcoords="offset points", ha='center', va='bottom')
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=90)

values4 = [772, 740, 753, 608, 816, 80, 160]
bars4 = ax4.bar(names, values4, color='darkorange')
for bar in bars4:
   height = bar.get_height()
   ax4.annotate(f'{height}', xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3),
   textcoords="offset points", ha='center', va='bottom')
plt.setp(ax4.xaxis.get_majorticklabels(), rotation=90)

fig.text(0.5, -0.05, 'Number of samples', ha='center', weight='bold')
fig.text(0.08, 0.5, 'Domain classes', va='center', rotation='vertical', weight='bold')
plt.suptitle('PACS Dataset', weight='bold')
plt.show()
