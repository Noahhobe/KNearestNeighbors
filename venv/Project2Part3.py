from scipy.io import arff
import scipy.spatial.distance as dist

#Inputs the training and testing data
training = raw_input("Enter in the name for the training file: ")
testing = raw_input("Enter in the name for the testing file: ")
trainingData = arff.loadarff(training)
testingData = arff.loadarff(testing)

#Gets value for k
# k = int(raw_input("Enter number for k amount of folds: "))
k = 3

#Sets some initial varaibles, testData and trainData get rid of the @attribute rows, leaving only the data
testData = testingData[0]
trainData = trainingData[0]
testRow = 0
dataLen = len(trainingData[0][0]) - 1
testLabel1 = trainingData[0][0][dataLen]



#Beginning of the KNN Algorithm
while k <= 11:
    print("==================================================")
    print("Values for K = {}".format(k))
    print("==================================================")
    truePosE = 0.0
    trueNegE = 0.0
    falsePosE = 0.0
    falseNegE = 0.0
    truePosM = 0.0
    trueNegM = 0.0
    falsePosM = 0.0
    falseNegM = 0.0
    truePosC = 0.0
    trueNegC = 0.0
    falsePosC = 0.0
    falseNegC = 0.0
    truePosCos = 0.0
    trueNegCos = 0.0
    falsePosCos = 0.0
    falseNegCos = 0.0
    for testGene in testData:
        #Initializes arrays of the various distances, they will contain distances between one test row and all training rows
        EdistList = []
        MdistList = []
        CdistList = []
        CosSimList = []
        trainRow = 0
        for trainGene in trainData:
            #Finds the second class label once it first appears in training data
            if(trainingData[0][trainRow][dataLen] != testLabel1):
                testLabel2 = trainingData[0][trainRow][dataLen]
            aCount = 0
            trainList = []
            testList = []
            #Creates two lists that are filled with the attributes of the current
            #training and testing tuple respectively
            while aCount < len(trainGene) - 1:
                trainList.append(trainGene[aCount])
                testList.append(testGene[aCount])
                aCount += 1
            #Calculates the four distance values between curreent train and test sample
            Edistance = dist.euclidean(trainList, testList)
            Mdistance = dist.cityblock(trainList, testList)
            Cdistance = dist.chebyshev(trainList, testList)
            CosineSim = 1 - dist.cosine(trainList, testList)

            #Distances are added to an array, will eventually contain all distances
            #between current test sample and all training samples
            EdistList.append((Edistance, trainRow))
            MdistList.append((Mdistance, trainRow))
            CdistList.append((Cdistance, trainRow))
            CosSimList.append((CosineSim, trainRow))

            #Increments counter for current row number training samples are on
            trainRow += 1

        #sorts all distance lists
        list.sort(EdistList)
        list.sort(MdistList)
        list.sort(CdistList)
        list.sort(CosSimList)

        #initializes counters for weighted distance of first and second class label
        #for all four different distance measures
        EfirstCount = 0.0
        EsecondCount = 0.0
        MfirstCount = 0.0
        MsecondCount = 0.0
        CfirstCount = 0.0
        CsecondCount = 0.0
        CosfirstCount = 0.0
        CossecondCount = 0.0
        for x in range (0, k):
            closeEdist = EdistList[x][0]
            closeMdist = MdistList[x][0]
            closeCdist = CdistList[x][0]
            closeCosdist = CosSimList[x][0]
            # Counts number of classifications from euclidean list with weighted distance
            if (testLabel1 == trainingData[0][EdistList[x][1]][dataLen]):
                EfirstCount += 1 * (1/((closeEdist)**2 + 1))
            else:
                EsecondCount += 1 * (1/((closeEdist)**2 + 1))
            # Counts number of classifications from Manhattan list with weighted distance
            if (testLabel1 == trainingData[0][MdistList[x][1]][dataLen]):
                MfirstCount += 1 * (1/((closeMdist)**2 + 1))
            else:
                MsecondCount += 1 * (1/((closeMdist)**2 + 1))
            # Counts number of classifications from chebyshev list with weighted distance
            if (testLabel1 == trainingData[0][CdistList[x][1]][dataLen]):
                CfirstCount += 1 * (1/((closeCdist)**2 + 1))
            else:
                CsecondCount += 1 * (1/((closeCdist)**2 + 1))
            # Counts number of classifications from cosine similarity list with weighted distance
            if (testLabel1 == trainingData[0][CosSimList[x][1]][dataLen]):
                CosfirstCount += 1 * (closeCosdist)**2
            else:
                CossecondCount += 1 * (closeCosdist)**2
        #This is the class prediction using k and euclidean disctance
        if (EfirstCount > EsecondCount):
            euclidPrediction = testLabel1
        else:
            euclidPrediction = testLabel2
        print("Class prediction using euclidean distance for test sample number {}: ".format(testRow + 1))
        print(euclidPrediction)
        #This is the class prediction using k and Manhattan disctance
        if (MfirstCount > MsecondCount):
            manhattanPrediction = testLabel1
        else:
            manhattanPrediction = testLabel2
        print("Class prediction using manhattan distance for test sample number {}: ".format(testRow + 1))
        print(manhattanPrediction)
        #This is the class prediction using k and chebyshev disctance
        if (CfirstCount > CsecondCount):
            chebyPrediction = testLabel1
        else:
            chebyPrediction = testLabel2
        print("Class prediction using chebyshev distance for test sample number {}: ".format(testRow + 1))
        print(chebyPrediction)
        #This is the class prediction using k and cosine similarity
        if (CosfirstCount > CossecondCount):
            cosinePrediction = testLabel1
        else:
            cosinePrediction = testLabel2
        print("Class prediction using cosine similarity for test sample number {}: ".format(testRow + 1))
        print(cosinePrediction)
        print("")

        #Gives the numbers for matrix diagrams
        if (euclidPrediction == testingData[0][testRow][dataLen] and euclidPrediction == testLabel1):
            truePosE += 1.0
        elif (euclidPrediction == testingData[0][testRow][dataLen] and euclidPrediction == testLabel2):
            trueNegE += 1.0
        elif(euclidPrediction != testingData[0][testRow][dataLen] and euclidPrediction == testLabel2):
            falseNegE += 1.0
        else:
            falsePosE += 1.0

        if (manhattanPrediction == testingData[0][testRow][dataLen] and manhattanPrediction == testLabel1):
            truePosM += 1.0
        elif (manhattanPrediction == testingData[0][testRow][dataLen] and manhattanPrediction == testLabel2):
            trueNegM += 1.0
        elif(manhattanPrediction != testingData[0][testRow][dataLen] and manhattanPrediction == testLabel2):
            falseNegM += 1.0
        else:
            falsePosM += 1.0

        if (chebyPrediction == testingData[0][testRow][dataLen] and chebyPrediction == testLabel1):
            truePosC += 1.0
        elif (chebyPrediction == testingData[0][testRow][dataLen] and chebyPrediction == testLabel2):
            trueNegC += 1.0
        elif(chebyPrediction != testingData[0][testRow][dataLen] and chebyPrediction == testLabel2):
            falseNegC += 1.0
        else:
            falsePosC += 1.0

        if (cosinePrediction == testingData[0][testRow][dataLen] and cosinePrediction == testLabel1):
            truePosCos += 1.0
        elif (cosinePrediction == testingData[0][testRow][dataLen] and cosinePrediction == testLabel2):
            trueNegCos += 1.0
        elif(cosinePrediction != testingData[0][testRow][dataLen] and cosinePrediction == testLabel2):
            falseNegCos += 1.0
        else:
            falsePosCos += 1.0
        #Increments counter for current row number training samples are on
        testRow += 1
    #Prints the the confusion matrix data and the precision, recall and F-1
    #values for every class label and distance measure combination 
    print("Confusion Matrix {} values for euclidean are:".format(testLabel1) )
    print("TP: {} FP: {} FN: {} TN: {}".format(truePosE, falsePosE, falseNegE, trueNegE))
    precision = truePosE / (truePosE + falsePosE)
    recall = truePosE / (truePosE + trueNegE)
    F1 = 2.0 * (precision * recall ) / (precision + recall)
    print("Precision: {} Recall: {} F-1: {}".format(precision, recall, F1))
    print("")

    print("Confusion Matrix {} values for manhattan are:".format(testLabel1) )
    print("TP: {} FP: {} FN: {} TN: {}".format(truePosM, falsePosM, falseNegM, trueNegM))
    precision = truePosM / (truePosM + falsePosM)
    recall = truePosM / (truePosM + trueNegM)
    F1 = 2.0 * (precision * recall) / (precision + recall)
    print("Precision: {} Recall: {} F-1: {}".format(precision, recall, F1))
    print("")

    print("Confusion Matrix {} values for chebyshev are:".format(testLabel1) )
    print("TP: {} FP: {} FN: {} TN: {}".format(truePosC, falsePosC, falseNegC, trueNegC))
    precision = truePosC / (truePosC + falsePosC)
    recall = truePosC / (truePosC + trueNegC)
    F1 = 2.0 * (precision * recall ) / (precision + recall)
    print("Precision: {} Recall: {} F-1: {}".format(precision, recall, F1))
    print("")

    print("Confusion Matrix {} values for cosine similarity are:".format(testLabel1) )
    print("TP: {} FP: {} FN: {} TN: {}".format(truePosCos, falsePosCos, falseNegCos, trueNegCos))
    precision = truePosCos / (truePosCos + falsePosCos)
    recall = truePosCos / (truePosCos + trueNegCos)
    F1 = 2.0 * (precision * recall ) / (precision + recall)
    print("Precision: {} Recall: {} F-1: {}".format(precision, recall, F1))
    print("")
    print("---------------------------------------------")
    print("Confusion Matrix {} values for euclidean are:".format(testLabel2) )
    print("TP: {} FP: {} FN: {} TN: {}".format(trueNegE, falseNegE, falsePosE, truePosE))
    precision = trueNegE / (trueNegE + falseNegE)
    recall = trueNegE / (trueNegE + truePosE)
    F1 = 2.0 * (precision * recall ) / (precision + recall)
    print("Precision: {} Recall: {} F-1: {}".format(precision, recall, F1))
    print("")

    print("Confusion Matrix {} values for manhattan are:".format(testLabel2))
    print("TP: {} FP: {} FN: {} TN: {}".format(trueNegM, falseNegM, falsePosM, truePosM))
    precision = trueNegM / (trueNegM + falseNegM)
    recall = trueNegM / (trueNegM + truePosM)
    F1 = 2.0 * (precision * recall ) / (precision + recall)
    print("Precision: {} Recall: {} F-1: {}".format(precision, recall, F1))
    print("")

    print("Confusion Matrix{} values for chebyshev are:".format(testLabel2) )
    print("TP: {} FP: {} FN: {} TN: {}".format(trueNegC, falseNegC, falsePosC, truePosC))
    precision = trueNegC / (trueNegC + falseNegC)
    recall = trueNegC / (trueNegC + truePosC)
    F1 = 2.0 * (precision * recall ) / (precision + recall)
    print("Precision: {} Recall: {} F-1: {}".format(precision, recall, F1))
    print("")

    print("Confusion Matrix {} values for cosine similarity are:".format(testLabel2) )
    print("TP: {} FP: {} FN: {} TN: {}".format(trueNegCos, falseNegCos, falsePosCos, truePosCos))
    precision = trueNegCos / (trueNegCos + falseNegCos)
    recall = trueNegCos / (trueNegCos + truePosCos)
    F1 = 2.0 * (precision * recall ) / (precision + recall)
    print("Precision: {} Recall: {} F-1: {}".format(precision, recall, F1))
    print("")
    #Increments K by 2
    k += 2
    testRow = 0
    trainRow = 0
