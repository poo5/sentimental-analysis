testFile = "bayes_best.py"
trainDir = "movies_reviews/"
testDir = "poo/"

execfile(testFile)
bc = Bayes_Classifier(trainDir)

iFileList = []

for fFileObj in os.walk(testDir + "/"):
    iFileList = fFileObj[2]
    break
print '%d test reviews.' % len(iFileList)

results = {"negative": 0, "neutral": 0, "positive": 0}

fpP = 0
fpN = 0
tpN = 0
tpP = 0

print "\nFile Classifications:"
for filename in iFileList:
    fileText = bc.loadFile(trainDir + filename)
    result = bc.classify(fileText)
    print "%s: %s" % (filename, result)
    results[result] += 1
    m = re.split("[-.]", filename)[-3]

    if m == '1' and result == "positive":
        fpP += 1

    if m == '1' and result == "negative":
        tpN += 1

    if m == '5' and result == "positive":
        tpP += 1

    if m == '5' and result == "negative":
        fpN += 1

precision_positive = tpP/float(tpP + fpP)
print precision_positive
precision_negative = tpN/float(tpN + fpN)
print precision_negative
recall_positive = tpP/float(tpP + fpN)
print recall_positive
recall_negative = tpN/float(tpN + fpP)
print recall_negative
f1_negative = 2*precision_negative*recall_negative/(recall_negative + precision_negative)
print f1_negative
f1_positive = 2*precision_positive*recall_positive/(recall_positive + precision_positive)
print f1_positive

print "\nResults Summary:"
for r in results:
    print "%s: %d" % (r, results[r])


