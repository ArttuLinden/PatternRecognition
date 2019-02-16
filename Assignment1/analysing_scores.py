import numpy as np
import matplotlib.pyplot as plt

all_scores = np.load('pca_comparison.npy')

#fft_scores = np.load('feature_comparison_fft.npy')
#all_scores = np.vstack((all_scores,fft_scores))


def getScores(all_scores,column):
    summary = []
    ids = np.unique(all_scores[:,column])
    for id in ids:
        mask = all_scores[:,column]==id
        glob_inds = np.where(mask)[0]
        scores = all_scores[mask,3]
        scores = [ float(x) for x in scores ]
        scores = np.array(scores)
        max_ind = np.argmax(scores)
        best = all_scores[glob_inds[max_ind],:]
        summary.append([id,best,scores])

    printSummary(summary,column)
    return summary

def plotSummary(summary):
    for thing in summary:
        plt.figure()
        plt.title("{}\nmean={:.5}\nstd={:.5}\nMax={:.5}\n({}, {}, {})".format(
                thing[0],np.mean(thing[2]),np.std(thing[2]),thing[1][3],
                thing[1][0],thing[1][1],thing[1][2]))
        plt.plot(thing[2])
        plt.plot([0,np.shape(thing[2])[0]],[np.mean(thing[2]),np.mean(thing[2])])
        
def printSummary(summary,ind):
    names = ['Classifier','Filter', 'Feature']
    print("\n\n{} Results:\n@    @    @    @".format(names[ind]))
    best_thing = 0
    for thing in summary:
        if best_thing==0:
            best_thing = thing
        elif np.mean(thing[2])>np.mean(best_thing[2]):
            best_thing = thing
                
        print()
        print(thing[0])
        print("Mean acc {:.5}, std {:.5}".format(np.mean(thing[2]),np.std(thing[2])))
        print("Max acc {:.5} with {},{},{}".format(
                thing[1][3],thing[1][0],thing[1][1],thing[1][2]))
    print("\n=================================================")
    print("Best {} on average:  {}\n(with {:.5} mean acc)".format(names[ind],best_thing[0],np.mean(best_thing[2])))
    print("=================================================")

def makeLatexTable(all_scores,column):
    scores = np.array(all_scores)

    sortInds = np.argsort(scores[:,3])[::-1]
    scores = scores[sortInds]
    with open('table.txt','w') as f:
        for score in scores:
            f.write("{:.6} & {} \\\\\n".format(score[3],score[column]))
    


#classifier_summary = getScores(all_scores,0)
#limit_method_summary = getScores(all_scores,1)
#feature_method_summary = getScores(all_scores,2)

makeLatexTable(all_scores,1)

#plotSummary(classifier_summary)
#plotSummary(limit_method_summary)
#plotSummary(feature_method_summary)