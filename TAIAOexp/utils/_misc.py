""" Miscellaneous utils """

from string import ascii_letters


def _generate_feature_names(nFeatures):
    """ Generates a list of length *nFeatures* with combinatinos of the abecedary. """

    if nFeatures > len(ascii_letters):
        featureNames = list()
        name, i, j = 0, 0, -1
        while name < nFeatures:
            if j == -1:
                fName = ascii_letters[i]
            else:
                fName = featureNames[j] + ascii_letters[i]
            featureNames.append(fName)
            i += 1
            if i % len(ascii_letters) == 0:
                j += 1
                i = 0
            name += 1
    else:
        featureNames = [ascii_letters[i] for i in range(nFeatures)]
    return featureNames
