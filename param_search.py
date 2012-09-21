import pandas

CV_SETS = 5
INTERESTING_FEATURES = ('cyls', 'displacement', 'hp',
                        'weight', 'acc', 'year', 'origin')

def get_features_and_labels(csv_file):
    df = pandas.read_csv(csv_file)
    labels = df['mpg'].values
    features = zip(*map(lambda f: df[f].values, INTERESTING_FEATURES))
    return features, labels


def cross_validation_sets(feature_vectors, labels, n=CV_SETS):
    feature_sets = [ [] for i in range(n) ]
    label_sets = [ [] for i in range(n) ]
    feature_it = iter(feature_vectors)
    label_it = iter(labels)
    i = 0
    try:
        while True:
            feature_sets[i].append(feature_it.next())
            label_sets[i].append(label_it.next())
            i = (i + 1) % n
    except:
        pass
    return feature_sets, label_sets


if __name__ == '__main__':
    feature_sets, label_sets = cross_validation_sets(*get_features_and_labels("auto-mpg-nameless.csv"))
    print feature_sets, label_sets
