import glob
import numpy as np
import scipy.sparse as sp

def read_glob(file_pattern, delimiter, secondary_delimiter, cell_feature=False):
    file_paths = glob.iglob(file_pattern)
    all_I = np.array([], dtype='int')
    all_J = np.array([], dtype='int')
    all_K = np.array([], dtype='int')
    all_values = np.array([])
    input_filenames = []
    for input_filename in sorted(file_paths):
        #print input_filename
        I = []
        J = []
        file_K = []
        file_values = []
        input_filenames.append(input_filename)
        infile = open(input_filename)
        categorical = False
        first = True
        for line in infile:
            if first:
                first = False
                if line.find(':') > -1:
                    categorical = True
                continue
            K = []
            values = []
            split_line = line.strip().split(delimiter)
            I.append(int(split_line[0]))

            if cell_feature:
                J.append(int(split_line[1]))
                features = split_line[2].split(secondary_delimiter)
            else:
                try:
                    features = split_line[1].split(secondary_delimiter)
                except IndexError:
                    features = [np.nan]
            if categorical:
                for feature in features:
                    feature_index, feature_value = feature.split(':')
                    K.append(int(feature_index))
                    values.append(bool(feature_value))
                file_K.append(K)
                file_values.append(values)
            else:
                for feature in features:
                    K.append(0)
                    values.append(float(feature))
                file_K.append(K)
                file_values.append(values)

        I = np.array(I, dtype='int').reshape((len(I), 1))
        try:
            all_I = np.hstack((all_I, I))
        except ValueError:
            all_I = I
            if len(all_I.shape) == 1:
                all_I = all_I.reshape((all_I.shape[0], 1))

        if cell_feature:
            J = np.array(J, dtype='int').reshape((len(J), 1))
            try:
                all_J = np.hstack((all_J, J))
            except ValueError:
                all_J = J
                if len(all_J.shape) == 1:
                    all_J = all_J.reshape((all_J.shape[0], 1))

        file_K = np.array(file_K, dtype='int')
        try:
            current_max_K = np.max(all_K)
        except ValueError, msg:
            current_max_K = 0
        try:
            all_K = np.hstack((all_K, file_K+current_max_K+1))
        except:
            all_K = file_K
            if len(all_K.shape) == 1:
                all_K = all_K.reshape((all_K.shape[0], 1))

        if categorical:
            sparse_I = np.array([(i-1)*np.ones(len(file_K[i-1])) for i in I]).ravel()
            sparse_K = np.array([category for category in file_K[i-1] for i in I]).ravel()
            sparse_values = np.array([value for value in file_values[i-1] for i in I]).ravel()
            print sparse_I.shape
            print sparse_K.shape
            print sparse_values.shape
            file_values = np.array(sp.coo_matrix((sparse_values, (sparse_I, sparse_K)), dtype='int').tocsr().todense(), dtype='int')
        else:
            file_values = np.array(file_values)
        try:
            all_values = np.hstack((all_values, file_values))
        except ValueError:
            all_values = file_values
            if len(all_values.shape) == 1:
                all_values = all_values.reshape((all_values.shape[0], 1))

    if not cell_feature:
        assert np.all(np.tile(all_I[:,0].reshape(len(all_I),1), (1,all_I.shape[1]))==all_I)
        I = all_I[:,0].reshape(len(all_I[:,0]), 1)
        return I, J, all_K, all_values, input_filenames
    else:
        #print file_pattern
        #print all_I.shape
        assert np.all(np.tile(all_I[:,0].reshape(len(all_I),1), (1,all_I.shape[1]))==all_I)
        assert np.all(np.tile(all_J[:,0].reshape(len(all_J),1), (1,all_J.shape[1]))==all_J)
        I = all_I[:,0].reshape(len(all_I[:,0]), 1)
        J = all_J[:,0].reshape(len(all_J[:,0]), 1)
        return I, J, all_K, all_values, input_filenames

def sanitize_cell_data(data):
    I = data[0]
    J = data[1]
    vals = data[2]
    for dimension in range(vals.shape[1]):
        valid_data_indices = np.isfinite(vals[:,dimension])
        valid_I = I[valid_data_indices]
        valid_J = J[valid_data_indices]
        valid_vals = vals[valid_data_indices,dimension]
        invalid_data_indices = np.isnan(vals[:,dimension])
        relevant_J = np.array(sorted(set(J[invalid_data_indices])))
        relevant_J_means = np.zeros(len(relevant_J))
        for j in relevant_J:
            j_indices = valid_J == j
            relevant_J_means[j] = np.mean(valid_vals[j_indices])
        vals[invalid_data_indices,dimension] = relevant_J_means[J[invalid_data_indices]]
    return I,J,vals

def sanitize_noncell_data(data):
    vals = data
    for dimension in range(vals.shape[1]):
        valid_data_indices = np.isfinite(vals[:,dimension])
        valid_vals = vals[valid_data_indices,dimension]
        invalid_data_indices = np.isnan(vals[:,dimension])
        vals[invalid_data_indices,dimension] = np.mean(valid_vals)
    return vals

def print_betas(betas):
    for i in range(betas.shape[0]):
        for j in range(betas.shape[1]):
            print "(%d,%d):" % (i,j)
            print "\t" + " ".join(map(lambda x: "%5.4f" % x, betas[i,j,:]))
        print

def roc_curve(test_preds, true_vals):
    #print true_vals.shape
    #print test_preds.shape
    #import matplotlib.pyplot as plt
    fn = np.zeros(len(set(test_preds)))
    tp = np.zeros(len(set(test_preds)))
    tpr = np.zeros(len(set(test_preds)))
    tn = np.zeros(len(set(test_preds)))
    fp = np.zeros(len(set(test_preds)))
    fpr = np.zeros(len(set(test_preds)))
    print "num unique thresholds: %d" % (len(set(test_preds)),)
    for i,thresh in enumerate(sorted(set(test_preds))):
        if i % 10000 == 0:
            print "unique threshold %d" % (i,)
        fn[i] = np.sum(true_vals[test_preds<thresh])
        tp[i] = np.sum(true_vals[test_preds>=thresh])
        tpr[i] = 1.0*(np.sum(true_vals[test_preds>=thresh]))/(np.sum(true_vals))
        tn[i] = np.sum(test_preds<thresh)-fn[i]
        fp[i] = np.sum(test_preds>=thresh)-tp[i]
        fpr[i] = 1.0*(np.sum(test_preds>=thresh)-tp[i])/(len(true_vals)-np.sum(true_vals)) if not (len(true_vals)-np.sum(true_vals)) == 0 else 0
    #plt.plot(tp/np.sum(true_vals), fp/(len(test_preds)-np.sum(true_vals)))
    #plt.show()
    data = {'fn':fn, 'tp':tp, 'tn':tn, 'fp':fp, 'tpr':tpr, 'fpr':fpr}
    import cPickle as pickle
    ofile = open('roc_output.txt', 'w')
    pickle.dump(data, ofile)
    ofile.close()
    return data

def auc(tpr, fpr):
    area = 0.0
    for i in range(len(tpr)):
        try:
            area += (fpr[i+1]-fpr[i])*(tpr[i]+tpr[i+1])/2.0
        except IndexError:
            return area
    return 0.0
