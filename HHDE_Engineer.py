import os
from copy import deepcopy
from enoppy.paper_based.pdo_2022 import *


PopSize = 100
DimSize = 10
LB = [-100] * DimSize
UB = [100] * DimSize
TrialRuns = 30
MaxFEs = 20000

MaxIter = int(MaxFEs / PopSize)
curIter = 0

Pop = np.zeros((PopSize, DimSize))
FitPop = np.zeros(PopSize)

Func_num = 0

BestIndi = None
FitBest = float("inf")


def Initial(func):
    global Pop, FitPop, DimSize, BestIndi, FitBest
    for i in range(PopSize):
        for j in range(DimSize):
            Pop[i][j] = LB[j] + (UB[j] - LB[j]) * np.random.rand()
        FitPop[i] = func.evaluate(Pop[i])
    FitBest = min(FitPop)
    BestIndi = deepcopy(Pop[np.argmin(FitPop)])


def HHDE(func):
    global Pop, curIter, MaxIter, LB, UB, PopSize, DimSize, BestIndi, FitBest

    Cr = 0.7
    Off = np.zeros((PopSize, DimSize))
    FitOff = np.zeros(PopSize)

    for i in range(PopSize):

        """ Mutation """
        idx_list = list(range(0, PopSize))
        R = np.random.rand()
        if R < 1 / 5:  # cur/1
            r1, r2 = np.random.choice(idx_list, 2, replace=False)
            Off[i] = Pop[i] + np.random.rand() * (Pop[r1] - Pop[r1])
        elif R < 2 / 5:  # rand/1
            r1, r2, r3 = np.random.choice(idx_list, 3, replace=False)
            Off[i] = Pop[r1] + np.random.rand() * (Pop[r2] - Pop[r3])
        elif R < 3 / 5:  # best/1
            r1, r2 = np.random.choice(idx_list, 2, replace=False)
            Off[i] = BestIndi + np.random.rand() * (Pop[r1] - Pop[r2])
        elif R < 4 / 5:  # cur-to-best/1
            idx_list.remove(i)
            r1, r2 = np.random.choice(idx_list, 2, replace=False)
            Off[i] = Pop[i] + np.random.rand() * (BestIndi - Pop[i]) + np.random.rand() * (Pop[r1] - Pop[r2])
        else:  # cur-to-pbest/1
            idx_sort = np.argsort(FitPop)
            pbest = np.mean(Pop[idx_sort[int(PopSize * 0.05)]], axis=0)
            r1, r2 = np.random.choice(idx_list, 2, replace=False)
            Off[i] = Pop[i] + np.random.rand() * (pbest - Pop[i]) + np.random.rand() * (Pop[r1] - Pop[r2])

        """ Crossover """
        R = np.random.rand()
        if R < 1 / 5:  # binary crossover with parent
            jrand = np.random.randint(0, DimSize)
            for j in range(DimSize):
                if np.random.rand() < Cr or j == jrand:
                    pass
                else:
                    Off[i][j] = Pop[i][j]
        elif R < 2 / 5:  # binary crossover with Xbest
            jrand = np.random.randint(0, DimSize)
            for j in range(DimSize):
                if np.random.rand() < Cr or j == jrand:
                    pass
                else:
                    Off[i][j] = BestIndi[j]
        elif R < 3 / 5:  # exponential crossover with parent
            jstart = np.random.randint(0, DimSize-1)
            while jstart < DimSize and np.random.rand() < Cr:
                Off[i][jstart] = Pop[i][jstart]
                jstart += 1
        elif R < 4 / 5:  # exponential crossover with parent
            jstart = np.random.randint(0, DimSize-1)
            while jstart < DimSize and np.random.rand() < Cr:
                Off[i][jstart] = BestIndi[jstart]
                jstart += 1
        else:
            jrand = np.random.randint(0, DimSize)
            for j in range(DimSize):
                if np.random.rand() < Cr or j == jrand:
                    pass
                else:
                    b = np.random.choice([0.1, 0.5, 0.9])
                    Off[i][j] = b * Pop[i][j] + (1 - b) * Off[i][j]

        """ Boundary repair """
        for j in range(DimSize):
            if Off[i][j] > UB[j] or Off[i][j] < LB[j]:
                R = np.random.rand()
                if R < 1 / 4:  # random
                    Off[i][j] = LB[j] + np.random.rand() * (UB[j] - LB[j])
                elif R < 2 / 4:  # bound place
                    if Off[i][j] > UB[j]:
                        Off[i][j] = UB[j]
                    else:
                        Off[i][j] = LB[j]
                elif R < 3 / 4:  # best inheritance
                    Off[i][j] = BestIndi[j]
                else:  # opposite learning
                    while Off[i][j] > UB[j] or Off[i][j] < LB[j]:
                        if Off[i][j] > UB[j]:
                            Off[i][j] = 2 * UB[j] - Off[i][j]
                        else:
                            Off[i][j] = 2 * LB[j] - Off[i][j]

        FitOff[i] = func.evaluate(Off[i])
        if FitOff[i] < FitPop[i]:
            FitPop[i] = FitOff[i]
            Pop[i] = deepcopy(Off[i])
            if FitOff[i] < FitBest:
                FitBest = FitOff[i]
                BestIndi = deepcopy(Off[i])


def RunHHDE(func):
    global curIter, MaxIter, TrialRuns, Pop, FitPop, DimSize
    All_Trial_Best = []
    for i in range(TrialRuns):
        Best_list = []
        curIter = 0
        Initial(func)
        Best_list.append(FitBest)
        np.random.seed(2022 + 88 * i)
        while curIter < MaxIter:
            HHDE(func)
            curIter += 1
            Best_list.append(FitBest)
        All_Trial_Best.append(Best_list)
    np.savetxt("./HHDE_Data/Engineer/" + str(Func_num) + ".csv", All_Trial_Best, delimiter=",")


def main():
    global Func_num, DimSize, Pop, MaxFEs, LB, UB

    MaxFEs = 20000

    Probs = [SRD(), TBTD(), GTD(), CBD(), CBHD(), RCB()]
    Names = ["SRD", "TBTD", "GTD", "CBD", "CBHD", "RCB"]

    for i in range(len(Probs)):
        DimSize = Probs[i].n_dims
        Pop = np.zeros((PopSize, DimSize))
        LB = Probs[i].lb
        UB = Probs[i].ub
        Func_num = Names[i]
        RunHHDE(Probs[i])


if __name__ == "__main__":
    if os.path.exists('./HHDE_Data/Engineer') == False:
        os.makedirs('./HHDE_Data/Engineer')
    main()
