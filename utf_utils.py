import re

def transform_german_word(word, lowercase=True):
    word = word.lower()
    word = re.sub('^ein\s|^eine\s|^der\s|^das\s|^die\s|^ne\s|^dann\s', '', word)
    word = re.sub('^e\s', 'e-', word)
    versions = [word]
    original_versions = [word]
    substitutions = [
                     ('ae', 'ä'),
                     ('oe', 'ö'),
                     ('ue', 'ü'),
                     ('ss', 'ß'),
                     ]
    for word in original_versions:
        ### collecting different versions of a word
        for forw, back in substitutions:
            if forw in word:
                new_versions = [w for w in versions]
                for w in new_versions:
                    corr_word = w.replace(forw, back)
                    versions.append(corr_word)
            if back in word:
                new_versions = [w for w in versions]
                for w in new_versions:
                    corr_word = w.replace(back, forw)
                    versions.append(corr_word)
    if not lowercase:
        versions = set(
                       ### capitalized
                       [' '.join([tok.capitalize() for tok in w.split()]) for w in versions] +\
                       [' '.join([tok.capitalize() for tok in word.split()])] + \
                       []
                       ### non-capitalized
                       #[w for w in versions]
                       )
    else:
        versions = set(
                       ### capitalized
                       [' '.join([tok.capitalize() for tok in w.split()]) for w in versions] +\
                       [' '.join([tok.capitalize() for tok in word.split()])] + \
                       ### non-capitalized
                       [w for w in versions]
                       )

    #print(versions)
    return versions
