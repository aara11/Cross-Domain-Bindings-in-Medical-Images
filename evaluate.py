import numpy as np
import hickle


def get_concept_encoding(i, j):
    file_name = './enc_concept_' + str(int(i / 10000)) + '.hkl'
    embedding = hickle.load(file_name)
    index_start = i % 10000
    index_end = j % 10000
    if (int(i / 10000) == int(j / 10000)):
        return embedding[index_start:index_end]
    else:
        return np.concatenate((embedding[index_start:], get_concept_encoding(((i / 10000) + 1) * 10000, j)), axis=0)


def eval_(gt, pred_values, thresh):
    pre = 0
    if pred_values >= thresh:
        pre = 1

    ct_tp = ct_tn = ct_fn = ct_fp = 0
    if (pre == 1):
        if (gt == 1):
            ct_tp += 1
        else:
            ct_fp += 1
    else:
        if (gt == 1):
            ct_fn += 1
        else:
            ct_tn += 1
    return [ct_tp, ct_tn, ct_fn, ct_fp]


if __name__ == '__main__':
    gt = get_concept_encoding(20000, 25000)
    pred_values = hickle.load('./out_model2.h5')
    rp = []
    aa = np.linspace(0, 0.4, 21)
    f1_curr = 0
    idx = -1
    for thr in range(21):
        print(thr)
        thresh = aa[thr]
        #        thresh=0.065
        lst = []
        f1 = []
        rep = []

        for concept in range(1000):
            prd = pred_values[:, 0, concept]
            for i in range(5000):
                gtt = gt[i][concept]
                pred = prd[i]
                [a, b, c, d] = eval_(gtt, pred, thresh)
                lst.append([a, b, c, d])
                final = np.array(lst)
        eval_rp = np.mean(final, axis=0)
        rp.append(eval_rp)
        prec = eval_rp[0] / (eval_rp[0] + eval_rp[3])
        recal = eval_rp[0] / (eval_rp[0] + eval_rp[1])
        f1_score = 2 * prec * recal / (prec + recal)
        print(thresh, f1_score, prec, recal)
        if (f1_score > f1_curr):
            f1_curr = f1_score
            idx = thr
        f1.append(f1_score)
    print(f1_score)
    print(aa[thr])
