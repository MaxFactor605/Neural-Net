# A simple ML model for handwrite digits recognition

Repository structure:

* Source directory - source code (dirty code)

 - Make-train.py - program for initialize train set
 - Train-set.db, Test-set.db - python shelve files, store a train set and test set
 - weights.db, python shelve file, store a NN weights
 - Main_NN.py - main program,
   1. TRAIN - if true run gradient descent with given paramers
   2. TEST - if true run test and print train cost, test cost and difference
   3. GRADIENT_CHECK - if true run gradient_check for applying a back propagation (there is everyting okay don't touch it, it's very slow)

   
For create your own weights delete or transfer weights.db (Programm create a new random weights).
Algorithm is not very good because of lack a train exemples (I'm tired to draw this digits) and my dumpness (as always), but it's can correctly recognize well drawn digits.
