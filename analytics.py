#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main(df,fig,ax):
    # Question 2
    percentage_words_female = (df['Number words female'] / (df ['Total words']-df['Number of words lead'])) * 100
    percentage_words_male = (df['Number words male'] / (df['Total words']-df['Number of words lead'])) * 100
    total = percentage_words_female + percentage_words_male
    labels = df['Year'] 
    ax[0].bar(labels, percentage_words_male,width=0.65, label='Female')
    ax[0].bar(labels,percentage_words_female,width=0.65,bottom=percentage_words_male,label='Male')
    ax[0].set_ylabel('% of words spoken')
    ax[0].set_ylim(top=100)
    ax[0].set_xlim(min(labels)-2, max(labels)+2)
    ax[0].legend()
    # -----------
    # Question 3
    male_gt_female = df[df['Number words male'] > df['Number words female']]
    female_gt_male = df[df['Number words female'] > df['Number words male']]
    avg_mgtf = np.average(male_gt_female.Gross)
    avg_fgtm = np.average(female_gt_male.Gross)
    ax[1].bar("Average Gross - M Words > F Words",avg_mgtf)
    ax[1].bar("Average Gross - F Words > M Words",avg_fgtm)
    ax[1].set_ylabel('Gross Revenue in 1000s')
    # -----------    
    plt.show()

if __name__ == "__main__":
    fig, ax = plt.subplots(2,1)
    train = pd.read_csv('data/train.csv')
    main(train,fig,ax)



